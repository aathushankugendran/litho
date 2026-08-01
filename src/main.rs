use llm_engine::{gguf, model, sampler, tokenizer};
use sampler::SamplingConfig;

fn main() {
    let path = "/Users/aathushankugendran/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    let file = gguf::parse(path).expect("failed to parse GGUF file");

    println!("Loading model...");
    let model = model::load_model(path, &file);
    let tok = tokenizer::load_tokenizer(&file);
    println!(
        "Loaded {} layers, hidden_size={}, vocab_size={}",
        model.config.n_layers, model.config.hidden_size, model.config.vocab_size
    );

    let prompt = "The capital of France is";
    let mut prompt_ids = vec![tok.bos_id()];
    prompt_ids.extend(tok.encode(prompt));

    println!("\nprompt: {prompt:?}");

    // --- Streaming cached generation, greedy ---
    // This loop is the terminal stand-in for what a live UI will eventually
    // render: each event fires the moment a token is produced, carrying
    // enough information to update a stats panel without waiting for the
    // whole response.
    println!("\n--- streaming cached generation (greedy) ---");
    let generated = model::generate_cached_streaming(&model, &prompt_ids, 15, None, tok.eos_id(), |event| {
        let piece = tok.decode(&[event.token_id]);
        println!(
            "step {:<3} token_id={:<6} elapsed={:>5}ms cache={:>6}KB piece={:?}",
            event.step,
            event.token_id,
            event.elapsed_ms,
            event.cache_bytes / 1024,
            piece
        );
    });
    println!("\nfull text: {:?}", tok.decode(&generated));

    // --- Streaming cached generation, sampled ---
    println!("\n--- streaming cached generation (sampled) ---");
    let sampling = SamplingConfig { temperature: 0.8, top_k: Some(40), top_p: Some(0.9) };
    let sampled = model::generate_cached_streaming(&model, &prompt_ids, 15, Some(&sampling), tok.eos_id(), |event| {
        let piece = tok.decode(&[event.token_id]);
        println!(
            "step {:<3} token_id={:<6} elapsed={:>5}ms cache={:>6}KB piece={:?}",
            event.step,
            event.token_id,
            event.elapsed_ms,
            event.cache_bytes / 1024,
            piece
        );
    });
    println!("\nfull text: {:?}", tok.decode(&sampled));

    // --- Streaming naive generation, for comparison ---
    // Notice cache_bytes stays 0 throughout -- naive generation never
    // builds a cache, which is exactly the point being demonstrated.
    println!("\n--- streaming naive generation (greedy) ---");
    let naive = model::generate_naive_streaming(&model, &prompt_ids, 15, tok.eos_id(), |event| {
        let piece = tok.decode(&[event.token_id]);
        println!(
            "step {:<3} token_id={:<6} elapsed={:>5}ms cache={:>6}KB piece={:?}",
            event.step,
            event.token_id,
            event.elapsed_ms,
            event.cache_bytes / 1024,
            piece
        );
    });
    println!("\nfull text: {:?}", tok.decode(&naive));
}