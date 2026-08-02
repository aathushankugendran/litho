// model.rs
//
// Implements a full Llama-style transformer forward pass over a sequence of
// tokens, using naive (full re-computation) causal self-attention. No
// KV-cache yet: every generation step recomputes attention over the entire
// sequence seen so far. This version exists to establish a correct,
// easy-to-verify baseline before we introduce the KV-cache optimization,
// which must produce identical output while doing less work.


use crate::gguf::{GgufFile, GgufValue, TensorInfo};
use crate::tensor::{self, matmul};
use crate::kv_cache::{KvCache, LayerCache};
use crate::sampler::{sample, SamplingConfig};
use std::time::Instant;
use crate::kv_cache::BatchKvCache;
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_q_dims: (usize, usize),
    pub attn_k: Vec<f32>,
    pub attn_k_dims: (usize, usize),
    pub attn_v: Vec<f32>,
    pub attn_v_dims: (usize, usize),
    pub attn_output: Vec<f32>,
    pub attn_output_dims: (usize, usize),
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_gate_dims: (usize, usize),
    pub ffn_up: Vec<f32>,
    pub ffn_up_dims: (usize, usize),
    pub ffn_down: Vec<f32>,
    pub ffn_down_dims: (usize, usize),
}

pub struct Config {
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
}

pub struct Model {
    pub config: Config,
    pub layers: Vec<LayerWeights>,
    pub token_embd: Vec<f32>,   // [vocab_size * hidden_size]
    pub output_norm: Vec<f32>,  // final RMSNorm weight
    pub output_weight: Vec<f32>,
    pub output_dims: (usize, usize),
}

fn find_tensor<'a>(file: &'a GgufFile, name: &str) -> &'a TensorInfo {
    file.tensors.iter().find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tensor not found: {name}"))
}

fn load(path: &str, file: &GgufFile, name: &str) -> Vec<f32> {
    let info = find_tensor(file, name);
    tensor::load_tensor(path, file.tensor_data_offset, info)
}

// GGUF stores 2D weights as [in_features, out_features], the reverse of
// PyTorch's [out, in] convention. Confirmed earlier against known layer
// sizes (e.g. ffn_gate.weight is [hidden_size, intermediate_size]).
fn dims2(file: &GgufFile, name: &str) -> (usize, usize) {
    let info = find_tensor(file, name);
    assert_eq!(info.dims.len(), 2, "expected a 2D weight tensor for {name}");
    (info.dims[0] as usize, info.dims[1] as usize)
}

fn load_layer(path: &str, file: &GgufFile, layer: usize) -> LayerWeights {
    let p = |suffix: &str| format!("blk.{layer}.{suffix}");
    LayerWeights {
        attn_norm: load(path, file, &p("attn_norm.weight")),
        attn_q: load(path, file, &p("attn_q.weight")),
        attn_q_dims: dims2(file, &p("attn_q.weight")),
        attn_k: load(path, file, &p("attn_k.weight")),
        attn_k_dims: dims2(file, &p("attn_k.weight")),
        attn_v: load(path, file, &p("attn_v.weight")),
        attn_v_dims: dims2(file, &p("attn_v.weight")),
        attn_output: load(path, file, &p("attn_output.weight")),
        attn_output_dims: dims2(file, &p("attn_output.weight")),
        ffn_norm: load(path, file, &p("ffn_norm.weight")),
        ffn_gate: load(path, file, &p("ffn_gate.weight")),
        ffn_gate_dims: dims2(file, &p("ffn_gate.weight")),
        ffn_up: load(path, file, &p("ffn_up.weight")),
        ffn_up_dims: dims2(file, &p("ffn_up.weight")),
        ffn_down: load(path, file, &p("ffn_down.weight")),
        ffn_down_dims: dims2(file, &p("ffn_down.weight")),
    }
}

fn load_config(file: &GgufFile) -> Config {
    let get_u32 = |key: &str| -> u32 {
        file.metadata.iter().find(|(k, _)| k == key)
            .and_then(|(_, v)| if let GgufValue::U32(n) = v { Some(*n) } else { None })
            .unwrap_or_else(|| panic!("missing metadata key: {key}"))
    };
    let get_f32 = |key: &str| -> f32 {
        file.metadata.iter().find(|(k, _)| k == key)
            .and_then(|(_, v)| if let GgufValue::F32(n) = v { Some(*n) } else { None })
            .unwrap_or_else(|| panic!("missing metadata key: {key}"))
    };

    let (_, vocab_size) = dims2(file, "token_embd.weight");

    Config {
        hidden_size: get_u32("llama.embedding_length") as usize,
        n_heads: get_u32("llama.attention.head_count") as usize,
        n_kv_heads: get_u32("llama.attention.head_count_kv") as usize,
        head_dim: get_u32("llama.rope.dimension_count") as usize,
        n_layers: get_u32("llama.block_count") as usize,
        vocab_size,
        rms_eps: get_f32("llama.attention.layer_norm_rms_epsilon"),
    }
}

// Loads every layer's weights plus the embedding table, final norm, and
// output projection. This is the entire model resident in memory at once —
// fine for a 1-3B model on a machine with enough RAM, though a production
// runtime would mmap this instead of copying it. That's a Milestone 7
// concern, not this one.
pub fn load_model(path: &str, file: &GgufFile) -> Model {
    let config = load_config(file);
    let layers = (0..config.n_layers)
        .map(|i| load_layer(path, file, i))
        .collect();

    Model {
        config,
        layers,
        token_embd: load(path, file, "token_embd.weight"),
        output_norm: load(path, file, "output_norm.weight"),
        output_weight: load(path, file, "output.weight"),
        output_dims: dims2(file, "output.weight"),
    }
}

// --- RMSNorm ---
// Rescales a vector by its root-mean-square magnitude, then applies a
// learned per-element gain. No mean subtraction, unlike LayerNorm — cheaper,
// and works just as well for transformers in practice.
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(v, w)| v * scale * w).collect()
}

// weight dims are (in, out); computes y = W * x for one vector.
fn linear(weight: &[f32], dims: (usize, usize), x: &[f32]) -> Vec<f32> {
    let (in_dim, out_dim) = dims;
    assert_eq!(x.len(), in_dim, "input dim mismatch for linear layer");
    matmul(weight, out_dim, in_dim, x, 1)
}

// --- RoPE ---
// Encodes token position by rotating pairs of elements within a head's
// vector. Pair i rotates at a frequency that decreases as i grows, so early
// dimensions capture fine-grained position and later ones capture coarse
// position. This lets attention scores implicitly depend on the relative
// distance between two tokens, without a separate learned position table.
fn apply_rope(x: &mut [f32], position: usize, head_dim: usize) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / 10000f32.powf((2 * i) as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin, cos) = angle.sin_cos();

        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// --- SwiGLU feed-forward block ---
// Two projections of the same input (gate, up); the gate branch passes
// through SiLU and elementwise-multiplies the up branch before a final
// down-projection. Outperforms a plain ReLU MLP at equal parameter count,
// which is why Llama-family models use it.
fn swiglu_mlp(x: &[f32], w: &LayerWeights) -> Vec<f32> {
    let gate = linear(&w.ffn_gate, w.ffn_gate_dims, x);
    let up = linear(&w.ffn_up, w.ffn_up_dims, x);
    let activated: Vec<f32> = gate.iter().zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect();
    linear(&w.ffn_down, w.ffn_down_dims, &activated)
}

fn softmax(scores: &mut [f32]) {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
}

// --- Causal grouped-query attention over a full sequence ---
// Recomputes Q, K, V for every position in `xs` from scratch on every call.
// This is the "naive" baseline: correct, but redundant across generation
// steps, since positions 0..n-1 never change once computed. The KV-cache
// version replaces this by storing K/V per position instead of recomputing
// them; both must produce identical output, since caching does not change
// the math, only how many times it's performed.
fn attention_naive(xs: &[Vec<f32>], w: &LayerWeights, cfg: &Config) -> Vec<Vec<f32>> {
    let seq_len = xs.len();
    let group_size = cfg.n_heads / cfg.n_kv_heads;

    // Project every position to Q, K, V and apply RoPE up front, since each
    // position's rotated Q/K depends only on its own value and position —
    // not on any other position in the sequence.
    let mut all_q = Vec::with_capacity(seq_len);
    let mut all_k = Vec::with_capacity(seq_len);
    let mut all_v = Vec::with_capacity(seq_len);

    for (pos, x) in xs.iter().enumerate() {
        let q = linear(&w.attn_q, w.attn_q_dims, x);
        let k = linear(&w.attn_k, w.attn_k_dims, x);
        let v = linear(&w.attn_v, w.attn_v_dims, x);

        let mut q_per_head: Vec<Vec<f32>> = (0..cfg.n_heads)
            .map(|h| q[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
            .collect();
        let mut k_per_head: Vec<Vec<f32>> = (0..cfg.n_kv_heads)
            .map(|h| k[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
            .collect();
        let v_per_head: Vec<Vec<f32>> = (0..cfg.n_kv_heads)
            .map(|h| v[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
            .collect();

        for qh in q_per_head.iter_mut() {
            apply_rope(qh, pos, cfg.head_dim);
        }
        for kh in k_per_head.iter_mut() {
            apply_rope(kh, pos, cfg.head_dim);
        }

        all_q.push(q_per_head);
        all_k.push(k_per_head);
        all_v.push(v_per_head);
    }

    // For each position, attend only over itself and earlier positions
    // (causal masking) — a token must never see the future during training
    // or generation.
    let mut outputs = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let mut head_outputs = vec![0f32; cfg.n_heads * cfg.head_dim];

        for h in 0..cfg.n_heads {
            let kv_h = h / group_size;
            let qh = &all_q[pos][h];

            let mut scores: Vec<f32> = (0..=pos)
                .map(|j| {
                    let kh = &all_k[j][kv_h];
                    let dot: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
                    dot / (cfg.head_dim as f32).sqrt()
                })
                .collect();

            softmax(&mut scores);

            let mut weighted_sum = vec![0f32; cfg.head_dim];
            for (j, &weight) in scores.iter().enumerate() {
                let vh = &all_v[j][kv_h];
                for d in 0..cfg.head_dim {
                    weighted_sum[d] += weight * vh[d];
                }
            }

            head_outputs[h * cfg.head_dim..(h + 1) * cfg.head_dim]
                .copy_from_slice(&weighted_sum);
        }

        outputs.push(linear(&w.attn_output, w.attn_output_dims, &head_outputs));
    }

    outputs
}

// One full transformer layer applied to every position in the sequence.
fn forward_layer(xs: &[Vec<f32>], w: &LayerWeights, cfg: &Config) -> Vec<Vec<f32>> {
    let normed: Vec<Vec<f32>> = xs.iter()
        .map(|x| rms_norm(x, &w.attn_norm, cfg.rms_eps))
        .collect();

    let attn_out = attention_naive(&normed, w, cfg);

    let residual1: Vec<Vec<f32>> = xs.iter().zip(attn_out.iter())
        .map(|(x, a)| x.iter().zip(a.iter()).map(|(v, av)| v + av).collect())
        .collect();

    let normed2: Vec<Vec<f32>> = residual1.iter()
        .map(|x| rms_norm(x, &w.ffn_norm, cfg.rms_eps))
        .collect();

    let mlp_out: Vec<Vec<f32>> = normed2.iter()
        .map(|x| swiglu_mlp(x, w))
        .collect();

    residual1.iter().zip(mlp_out.iter())
        .map(|(r, m)| r.iter().zip(m.iter()).map(|(v, mv)| v + mv).collect())
        .collect()
}

fn embed_token(model: &Model, token_id: usize) -> Vec<f32> {
    let h = model.config.hidden_size;
    model.token_embd[token_id * h..(token_id + 1) * h].to_vec()
}

// Runs the full stack of layers on a token sequence and returns logits
// (unnormalized scores over the vocabulary) for the final position only —
// that's the one we need to pick the next token during generation.
pub fn forward(model: &Model, token_ids: &[usize]) -> Vec<f32> {
    let mut xs: Vec<Vec<f32>> = token_ids.iter()
        .map(|&id| embed_token(model, id))
        .collect();

    for layer in &model.layers {
        xs = forward_layer(&xs, layer, &model.config);
    }

    let last = xs.last().expect("empty sequence");
    let normed = rms_norm(last, &model.output_norm, model.config.rms_eps);
    linear(&model.output_weight, model.output_dims, &normed)
}

// Picks the single highest-scoring token. Deterministic — same input always
// produces the same output, which is exactly what we want for a correctness
// baseline. Temperature/top-k/top-p sampling comes in Milestone 5.
pub fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap()
}

// Naive autoregressive generation: on every step, re-run the entire
// sequence-so-far through the full model. Correct, but wasteful — this is
// the baseline the KV-cache version must match exactly while doing less
// redundant work.
pub fn generate_naive(model: &Model, prompt_ids: &[usize], n_new_tokens: usize) -> Vec<usize> {
    let mut tokens = prompt_ids.to_vec();

    for step in 0..n_new_tokens {
        let logits = forward(model, &tokens);
        let next_token = argmax(&logits);
        tokens.push(next_token);
        println!("  step {}/{n_new_tokens}: generated token {next_token}", step + 1);
    }

    tokens
}

// --- Causal grouped-query attention for a single new token, using a KV-cache ---
//
// Compare this to attention_naive, which loops over every position in the
// whole sequence and recomputes Q, K, and V for all of them on every call.
// Here, only the one new position (`x`) gets a fresh Q, K, and V computed.
// The new K and V are appended to the cache, and the new Q is compared
// against every K in the cache -- the ones computed just now, and every one
// computed on earlier calls. The math for any given position is identical
// to attention_naive; only the amount of repeated work differs.
fn attention_cached(
    x: &[f32],
    w: &LayerWeights,
    cfg: &Config,
    cache: &mut LayerCache,
    pos: usize,
) -> Vec<f32> {
    let group_size = cfg.n_heads / cfg.n_kv_heads;

    let q = linear(&w.attn_q, w.attn_q_dims, x);
    let k = linear(&w.attn_k, w.attn_k_dims, x);
    let v = linear(&w.attn_v, w.attn_v_dims, x);

    let mut q_per_head: Vec<Vec<f32>> = (0..cfg.n_heads)
        .map(|h| q[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
        .collect();
    let mut k_per_head: Vec<Vec<f32>> = (0..cfg.n_kv_heads)
        .map(|h| k[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
        .collect();
    let v_per_head: Vec<Vec<f32>> = (0..cfg.n_kv_heads)
        .map(|h| v[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
        .collect();

    for qh in q_per_head.iter_mut() {
        apply_rope(qh, pos, cfg.head_dim);
    }
    for kh in k_per_head.iter_mut() {
        apply_rope(kh, pos, cfg.head_dim);
    }

    // This push is the entire point of the cache: append the new position's
    // K/V, never recompute an old position's K/V.
    cache.keys.push(k_per_head);
    cache.values.push(v_per_head);

    let mut head_outputs = vec![0f32; cfg.n_heads * cfg.head_dim];

    for h in 0..cfg.n_heads {
        let kv_h = h / group_size;
        let qh = &q_per_head[h];

        // Attend over every cached position up to and including this one --
        // same causal restriction as attention_naive, just against a cache
        // instead of a freshly recomputed list.
        let mut scores: Vec<f32> = (0..=pos)
            .map(|j| {
                let kh = &cache.keys[j][kv_h];
                let dot: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
                dot / (cfg.head_dim as f32).sqrt()
            })
            .collect();

        softmax(&mut scores);

        let mut weighted_sum = vec![0f32; cfg.head_dim];
        for (j, &weight) in scores.iter().enumerate() {
            let vh = &cache.values[j][kv_h];
            for d in 0..cfg.head_dim {
                weighted_sum[d] += weight * vh[d];
            }
        }

        head_outputs[h * cfg.head_dim..(h + 1) * cfg.head_dim]
            .copy_from_slice(&weighted_sum);
    }

    linear(&w.attn_output, w.attn_output_dims, &head_outputs)
}

// One transformer layer applied to a single new token, reading from and
// writing to this layer's cache. Structurally identical to forward_layer,
// just operating on one hidden-state vector instead of a whole sequence.
fn forward_layer_cached(
    x: &[f32],
    w: &LayerWeights,
    cfg: &Config,
    cache: &mut LayerCache,
    pos: usize,
) -> Vec<f32> {
    let normed = rms_norm(x, &w.attn_norm, cfg.rms_eps);
    let attn_out = attention_cached(&normed, w, cfg, cache, pos);
    let residual1: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(v, a)| v + a).collect();

    let normed2 = rms_norm(&residual1, &w.ffn_norm, cfg.rms_eps);
    let mlp_out = swiglu_mlp(&normed2, w);
    residual1.iter().zip(mlp_out.iter()).map(|(v, m)| v + m).collect()
}

// Runs a single token through every layer of the model, using and updating
// the KV-cache at each layer. Returns logits for this one position, which is
// all generation ever needs -- the score distribution over the vocabulary
// for whatever token comes next.
pub fn forward_cached(model: &Model, cache: &mut KvCache, token_id: usize, pos: usize) -> Vec<f32> {
    let mut x = embed_token(model, token_id);

    for (layer, layer_cache) in model.layers.iter().zip(cache.layers.iter_mut()) {
        x = forward_layer_cached(&x, layer, &model.config, layer_cache, pos);
    }

    let normed = rms_norm(&x, &model.output_norm, model.config.rms_eps);
    linear(&model.output_weight, model.output_dims, &normed)
}

// Autoregressive generation using the KV-cache. Every prompt token is fed
// through once, in order, to populate the cache; each newly generated token
// is then fed through one at a time, reusing every previously cached K/V
// instead of recomputing them. This must produce identical token IDs to
// generate_naive given the same inputs -- the cache changes how much work
// gets repeated, never what the model actually computes.
pub fn generate_cached(model: &Model, prompt_ids: &[usize], n_new_tokens: usize) -> Vec<usize> {
    let mut cache = KvCache::new(model.config.n_layers);
    let mut tokens = prompt_ids.to_vec();
    let mut logits = Vec::new();

    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        logits = forward_cached(model, &mut cache, token_id, pos);
    }

    for step in 0..n_new_tokens {
        let next_token = argmax(&logits);
        tokens.push(next_token);

        let pos = prompt_ids.len() + step;
        logits = forward_cached(model, &mut cache, next_token, pos);
    }

    tokens
}

// Same as generate_cached, but picks each next token via a configurable
// sampling strategy instead of always taking the single highest-scoring
// token. This is what makes generated text feel varied instead of robotic.
pub fn generate_cached_sampled(
    model: &Model,
    prompt_ids: &[usize],
    n_new_tokens: usize,
    sampling: &SamplingConfig,
) -> Vec<usize> {
    let mut cache = KvCache::new(model.config.n_layers);
    let mut tokens = prompt_ids.to_vec();
    let mut logits = Vec::new();

    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        logits = forward_cached(model, &mut cache, token_id, pos);
    }

    for step in 0..n_new_tokens {
        let next_token = sample(&logits, sampling);
        tokens.push(next_token);

        let pos = prompt_ids.len() + step;
        logits = forward_cached(model, &mut cache, next_token, pos);
    }

    tokens
}

// One generated token's worth of progress information, handed to the
// caller's callback after every step. This is the data a live UI needs to
// update its stats panel and append text without waiting for the whole
// generation to finish.
pub struct TokenEvent {
    pub step: usize,
    pub token_id: usize,
    pub elapsed_ms: u128,
    pub cache_bytes: usize, // 0 for naive generation, which has no cache
}

// Same behavior as generate_naive, but invokes `on_token` after every
// generated token instead of only returning a result at the very end, and
// stops early if the model produces its own end-of-sequence token instead
// of always running for the full requested length regardless.
pub fn generate_naive_streaming(
    model: &Model,
    prompt_ids: &[usize],
    n_new_tokens: usize,
    eos_id: usize,
    mut on_token: impl FnMut(TokenEvent),
) -> Vec<usize> {
    let start = Instant::now();
    let mut tokens = prompt_ids.to_vec();

    for step in 0..n_new_tokens {
        let logits = forward(model, &tokens);
        let next_token = argmax(&logits);
        tokens.push(next_token);

        on_token(TokenEvent {
            step: step + 1,
            token_id: next_token,
            elapsed_ms: start.elapsed().as_millis(),
            cache_bytes: 0,
        });

        if next_token == eos_id {
            break;
        }
    }

    tokens
}

// Same behavior as generate_cached / generate_cached_sampled, but streams
// per-token progress through a callback, and stops early on an
// end-of-sequence token rather than always running the full requested
// length. Passing `sampling: None` behaves like greedy argmax; `Some(config)`
// samples using that configuration. Also reports the KV-cache's real memory
// footprint after every step, so a live UI can show it growing in real time.
pub fn generate_cached_streaming(
    model: &Model,
    prompt_ids: &[usize],
    n_new_tokens: usize,
    sampling: Option<&SamplingConfig>,
    eos_id: usize,
    mut on_token: impl FnMut(TokenEvent),
) -> Vec<usize> {
    let start = Instant::now();
    let mut cache = KvCache::new(model.config.n_layers);
    let mut tokens = prompt_ids.to_vec();
    let mut logits = Vec::new();

    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        logits = forward_cached(model, &mut cache, token_id, pos);
    }

    for step in 0..n_new_tokens {
        let next_token = match sampling {
            Some(cfg) => sample(&logits, cfg),
            None => argmax(&logits),
        };
        tokens.push(next_token);

        let pos = prompt_ids.len() + step;
        logits = forward_cached(model, &mut cache, next_token, pos);

        let cache_bytes = cache.memory_bytes(model.config.n_kv_heads, model.config.head_dim);
        on_token(TokenEvent {
            step: step + 1,
            token_id: next_token,
            elapsed_ms: start.elapsed().as_millis(),
            cache_bytes,
        });

        if next_token == eos_id {
            break;
        }
    }

    tokens
}
// --- Batched inference ---
//
// The reason batching helps is memory bandwidth, not arithmetic. Generating
// one token requires reading all ~1.1B weights out of memory and using each
// one exactly once, so the processor spends most of its time waiting on
// memory rather than computing. Batching several sequences together reads
// those same weights once but uses each one `batch_size` times, amortizing
// the expensive part across more useful work.
//
// Concretely, every projection changes from `W * one_vector` to
// `W * matrix_of_batch_vectors` -- the same matmul with a wider right-hand
// side, which is why matmul was written with an `n` parameter from the start.

// A batch of hidden states, stored column-major: element (row, seq) lives at
// data[row * batch_size + seq]. This layout is what matmul expects for its
// right-hand operand, so batched projections need no repacking.
pub struct BatchState {
    pub data: Vec<f32>,
    pub rows: usize,
    pub batch_size: usize,
}

impl BatchState {
    fn from_vectors(vectors: &[Vec<f32>]) -> Self {
        let batch_size = vectors.len();
        let rows = vectors[0].len();
        let mut data = vec![0f32; rows * batch_size];
        for (seq, v) in vectors.iter().enumerate() {
            for (row, &value) in v.iter().enumerate() {
                data[row * batch_size + seq] = value;
            }
        }
        BatchState { data, rows, batch_size }
    }

    fn column(&self, seq: usize) -> Vec<f32> {
        (0..self.rows).map(|row| self.data[row * self.batch_size + seq]).collect()
    }

    fn set_column(&mut self, seq: usize, values: &[f32]) {
        for (row, &value) in values.iter().enumerate() {
            self.data[row * self.batch_size + seq] = value;
        }
    }
}

// Batched linear projection: one matmul covering every sequence at once,
// rather than `batch_size` separate matmuls. This is where the bandwidth
// amortization actually happens -- the weight matrix is read once and
// applied across all columns.
fn linear_batched(weight: &[f32], dims: (usize, usize), x: &BatchState) -> BatchState {
    let (in_dim, out_dim) = dims;
    assert_eq!(x.rows, in_dim, "input dim mismatch for batched linear layer");

    let data = matmul(weight, out_dim, in_dim, &x.data, x.batch_size);
    BatchState { data, rows: out_dim, batch_size: x.batch_size }
}

fn rms_norm_batched(x: &BatchState, weight: &[f32], eps: f32) -> BatchState {
    let mut out = BatchState {
        data: vec![0f32; x.rows * x.batch_size],
        rows: x.rows,
        batch_size: x.batch_size,
    };
    for seq in 0..x.batch_size {
        let normed = rms_norm(&x.column(seq), weight, eps);
        out.set_column(seq, &normed);
    }
    out
}

fn swiglu_mlp_batched(x: &BatchState, w: &LayerWeights) -> BatchState {
    let gate = linear_batched(&w.ffn_gate, w.ffn_gate_dims, x);
    let up = linear_batched(&w.ffn_up, w.ffn_up_dims, x);

    let activated = BatchState {
        data: gate.data.iter().zip(up.data.iter()).map(|(g, u)| silu(*g) * u).collect(),
        rows: gate.rows,
        batch_size: gate.batch_size,
    };

    linear_batched(&w.ffn_down, w.ffn_down_dims, &activated)
}

// Attention is the one part that cannot be fully batched into a single
// matmul here: each sequence attends over its own distinct history, so the
// score computation and weighted sum run per sequence against that
// sequence's own cache. The Q/K/V projections that feed it are still fully
// batched, which is where most of the weight-reading cost lives.
fn attention_batched(
    x: &BatchState,
    w: &LayerWeights,
    cfg: &Config,
    caches: &mut [&mut LayerCache],
    positions: &[usize],
) -> BatchState {
    let group_size = cfg.n_heads / cfg.n_kv_heads;

    let q_all = linear_batched(&w.attn_q, w.attn_q_dims, x);
    let k_all = linear_batched(&w.attn_k, w.attn_k_dims, x);
    let v_all = linear_batched(&w.attn_v, w.attn_v_dims, x);

    let mut head_outputs = BatchState {
        data: vec![0f32; cfg.n_heads * cfg.head_dim * x.batch_size],
        rows: cfg.n_heads * cfg.head_dim,
        batch_size: x.batch_size,
    };

    for seq in 0..x.batch_size {
        let pos = positions[seq];
        let q = q_all.column(seq);
        let k = k_all.column(seq);
        let v = v_all.column(seq);

        let mut q_per_head: Vec<Vec<f32>> = (0..cfg.n_heads)
            .map(|h| q[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
            .collect();
        let mut k_per_head: Vec<Vec<f32>> = (0..cfg.n_kv_heads)
            .map(|h| k[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
            .collect();
        let v_per_head: Vec<Vec<f32>> = (0..cfg.n_kv_heads)
            .map(|h| v[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec())
            .collect();

        for qh in q_per_head.iter_mut() {
            apply_rope(qh, pos, cfg.head_dim);
        }
        for kh in k_per_head.iter_mut() {
            apply_rope(kh, pos, cfg.head_dim);
        }

        caches[seq].keys.push(k_per_head);
        caches[seq].values.push(v_per_head);

        let mut seq_out = vec![0f32; cfg.n_heads * cfg.head_dim];
        for h in 0..cfg.n_heads {
            let kv_h = h / group_size;
            let qh = &q_per_head[h];

            let mut scores: Vec<f32> = (0..=pos)
                .map(|j| {
                    let kh = &caches[seq].keys[j][kv_h];
                    let dot: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
                    dot / (cfg.head_dim as f32).sqrt()
                })
                .collect();

            softmax(&mut scores);

            let mut weighted_sum = vec![0f32; cfg.head_dim];
            for (j, &weight) in scores.iter().enumerate() {
                let vh = &caches[seq].values[j][kv_h];
                for d in 0..cfg.head_dim {
                    weighted_sum[d] += weight * vh[d];
                }
            }

            seq_out[h * cfg.head_dim..(h + 1) * cfg.head_dim].copy_from_slice(&weighted_sum);
        }

        head_outputs.set_column(seq, &seq_out);
    }

    linear_batched(&w.attn_output, w.attn_output_dims, &head_outputs)
}

fn forward_layer_batched(
    x: &BatchState,
    w: &LayerWeights,
    cfg: &Config,
    caches: &mut [&mut LayerCache],
    positions: &[usize],
) -> BatchState {
    let normed = rms_norm_batched(x, &w.attn_norm, cfg.rms_eps);
    let attn_out = attention_batched(&normed, w, cfg, caches, positions);

    let residual1 = BatchState {
        data: x.data.iter().zip(attn_out.data.iter()).map(|(a, b)| a + b).collect(),
        rows: x.rows,
        batch_size: x.batch_size,
    };

    let normed2 = rms_norm_batched(&residual1, &w.ffn_norm, cfg.rms_eps);
    let mlp_out = swiglu_mlp_batched(&normed2, w);

    BatchState {
        data: residual1.data.iter().zip(mlp_out.data.iter()).map(|(a, b)| a + b).collect(),
        rows: residual1.rows,
        batch_size: residual1.batch_size,
    }
}

// Runs one token per sequence through the whole model, returning logits for
// each sequence in the batch.
fn forward_batched(
    model: &Model,
    cache: &mut BatchKvCache,
    token_ids: &[usize],
    positions: &[usize],
) -> Vec<Vec<f32>> {
    let embeddings: Vec<Vec<f32>> = token_ids.iter().map(|&id| embed_token(model, id)).collect();
    let mut x = BatchState::from_vectors(&embeddings);

    for layer_idx in 0..model.config.n_layers {
        let mut layer_caches: Vec<&mut LayerCache> = cache
            .sequences
            .iter_mut()
            .map(|seq_cache| &mut seq_cache.layers[layer_idx])
            .collect();

        x = forward_layer_batched(
            &x,
            &model.layers[layer_idx],
            &model.config,
            &mut layer_caches,
            positions,
        );
    }

    (0..x.batch_size)
        .map(|seq| {
            let normed = rms_norm(&x.column(seq), &model.output_norm, model.config.rms_eps);
            linear(&model.output_weight, model.output_dims, &normed)
        })
        .collect()
}

// Batched autoregressive generation. All sequences advance in lockstep: one
// token per sequence per step. Prompts are padded to equal length by
// truncating to the shortest, which keeps the batch rectangular -- a real
// serving system would instead use continuous batching to swap finished
// sequences out and new ones in without stalling the batch.
pub fn generate_batched(
    model: &Model,
    prompts: &[Vec<usize>],
    n_new_tokens: usize,
) -> Vec<Vec<usize>> {
    let batch_size = prompts.len();
    let mut cache = BatchKvCache::new(batch_size, model.config.n_layers);

    let prompt_len = prompts.iter().map(|p| p.len()).min().expect("empty batch");
    let mut sequences: Vec<Vec<usize>> = prompts.iter().map(|p| p[..prompt_len].to_vec()).collect();

    let mut logits: Vec<Vec<f32>> = Vec::new();

    for pos in 0..prompt_len {
        let token_ids: Vec<usize> = sequences.iter().map(|s| s[pos]).collect();
        let positions = vec![pos; batch_size];
        logits = forward_batched(model, &mut cache, &token_ids, &positions);
    }

    for step in 0..n_new_tokens {
        let next_tokens: Vec<usize> = logits.iter().map(|l| argmax(l)).collect();
        for (seq, &tok) in next_tokens.iter().enumerate() {
            sequences[seq].push(tok);
        }

        let pos = prompt_len + step;
        let positions = vec![pos; batch_size];
        logits = forward_batched(model, &mut cache, &next_tokens, &positions);
    }

    sequences
}

// Total KV-cache memory across every sequence in a batch.
pub fn batch_cache_bytes(cache: &BatchKvCache, cfg: &Config) -> usize {
    cache.memory_bytes(cfg.n_kv_heads, cfg.head_dim)
}