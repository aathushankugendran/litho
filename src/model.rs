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