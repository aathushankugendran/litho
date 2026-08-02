// server.rs
//
// A small HTTP server wrapping the inference engine so a browser (or curl)
// can drive generation over the network instead of only through the CLI.
// The model loads once at startup and stays resident in memory across every
// request. Generation streams back token-by-token over Server-Sent Events,
// so a frontend can show text and stats updating live instead of waiting
// for the entire response.

use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use llm_engine::{gguf, model, sampler, tokenizer};
use model::Model;
use sampler::SamplingConfig;
use tokenizer::Tokenizer;

// Maps a GGML tensor dtype id to its human-readable quantization name.
// Only covers the types this project actually implements dequantization
// for; anything else is reported honestly as "unknown" rather than guessed.
fn dtype_name(dtype: u32) -> &'static str {
    match dtype {
        0 => "F32",
        1 => "F16",
        8 => "Q8_0",
        _ => "unknown",
    }
}

// Shared, read-only state every request has access to. Wrapped in an Arc so
// multiple concurrent requests can all read the same in-memory model
// without copying it.
struct AppState {
    model: Model,
    tok: Tokenizer,
    model_name: String,
    quantization: String,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    quantization: String,
    n_layers: usize,
    hidden_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    mode: String, // "naive" or "cached"
    max_tokens: usize,
    sampling: Option<SamplingRequest>,
}

#[derive(Deserialize)]
struct SamplingRequest {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

// One event sent to the browser per generated token. Mirrors model::TokenEvent
// but adds the decoded text piece, since the frontend shouldn't need its own
// copy of tokenizer logic just to display output.
#[derive(Serialize, Clone)]
struct StreamEvent {
    step: usize,
    token_id: usize,
    piece: String,
    elapsed_ms: u128,
    cache_kb: usize,
}

#[tokio::main]
async fn main() {
    let path = "/Users/aathushankugendran/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";

    println!("Loading model...");
    let file = gguf::parse(path).expect("failed to parse GGUF file");

    // Pull the model's display name straight from GGUF metadata rather than
    // hardcoding it, so this stays accurate if the model file changes.
    let model_name = file.metadata.iter()
        .find(|(k, _)| k == "general.name")
        .and_then(|(_, v)| if let gguf::GgufValue::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "unknown".to_string());

    // Quantization is read off one representative weight tensor -- most
    // Llama-style GGUF checkpoints quantize all large matrices the same way.
    let quantization = file.tensors.iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .map(|t| dtype_name(t.dtype).to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let model = model::load_model(path, &file);
    let tok = tokenizer::load_tokenizer(&file);
    println!(
        "Loaded {} layers, hidden_size={}, vocab_size={}",
        model.config.n_layers, model.config.hidden_size, model.config.vocab_size
    );

    let state = Arc::new(AppState { model, tok, model_name, quantization });

    let app = Router::new()
        .route("/model-info", get(model_info))
        .route("/generate", post(generate))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    let cfg = &state.model.config;
    Json(ModelInfo {
        name: state.model_name.clone(),
        quantization: state.quantization.clone(),
        n_layers: cfg.n_layers,
        hidden_size: cfg.hidden_size,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        vocab_size: cfg.vocab_size,
    })
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);

    // Generation is CPU-bound and synchronous, so it runs on a dedicated
    // blocking thread instead of the async runtime -- otherwise a single
    // long generation would stall every other request the server is
    // handling. Each token's progress is forwarded through the channel the
    // moment it's produced.
    tokio::task::spawn_blocking(move || {
        let mut prompt_ids = vec![state.tok.bos_id()];
        prompt_ids.extend(state.tok.encode(&req.prompt));
        let eos_id = state.tok.eos_id();

        let sampling_cfg = req.sampling.as_ref().map(|s| SamplingConfig {
            temperature: s.temperature,
            top_k: s.top_k,
            top_p: s.top_p,
        });

        // decode_piece preserves the leading space marker on each token
        // individually, unlike decode(), which trims whole-string leading
        // whitespace -- necessary here since tokens arrive one at a time as
        // fragments of an in-progress sentence, not as a complete string.
        let emit = |event: model::TokenEvent| {
            let piece = state.tok.decode_piece(event.token_id);
            let _ = tx.blocking_send(StreamEvent {
                step: event.step,
                token_id: event.token_id,
                piece,
                elapsed_ms: event.elapsed_ms,
                cache_kb: event.cache_bytes / 1024,
            });
        };

        if req.mode == "naive" {
            model::generate_naive_streaming(&state.model, &prompt_ids, req.max_tokens, eos_id, emit);
        } else {
            model::generate_cached_streaming(
                &state.model,
                &prompt_ids,
                req.max_tokens,
                sampling_cfg.as_ref(),
                eos_id,
                emit,
            );
        }
    });

    let stream = ReceiverStream::new(rx).map(|event| {
        let json = serde_json::to_string(&event).unwrap();
        Ok(Event::default().data(json))
    });

    Sse::new(stream)
}