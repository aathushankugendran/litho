// model.rs
//
// Implements one Llama-style transformer layer applied to a single token at
// position 0 (no KV-cache yet — that's Milestone 3). The goal here is to get
// every piece of math right and independently verifiable: RMSNorm, RoPE,
// grouped-query attention, and the SwiGLU feed-forward block.

use crate::gguf::{GgufFile, GgufValue, TensorInfo};
use crate::tensor::{self, matmul};

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_q_dims: (usize, usize), // (in, out)
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
    pub hidden_size: usize, // embedding_length
    pub n_heads: usize,     // attention.head_count
    pub n_kv_heads: usize,  // attention.head_count_kv
    pub head_dim: usize,    // rope.dimension_count
    pub rms_eps: f32,
}

fn find_tensor<'a>(file: &'a GgufFile, name: &str) -> &'a TensorInfo {
    file.tensors.iter().find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tensor not found: {name}"))
}

fn load(path: &str, file: &GgufFile, name: &str) -> Vec<f32> {
    let info = find_tensor(file, name);
    tensor::load_tensor(path, file.tensor_data_offset, info)
}

// GGUF stores weight dims as [in_features, out_features] — the reverse of
// PyTorch's usual [out, in] convention. Confirmed against known model sizes
// (e.g. ffn_gate.weight is [2048, 5632]: hidden_size in, intermediate out).
fn dims2(file: &GgufFile, name: &str) -> (usize, usize) {
    let info = find_tensor(file, name);
    assert_eq!(info.dims.len(), 2, "expected a 2D weight tensor for {name}");
    (info.dims[0] as usize, info.dims[1] as usize)
}

pub fn load_layer(path: &str, file: &GgufFile, layer: usize) -> LayerWeights {
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

pub fn load_config(file: &GgufFile) -> Config {
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

    Config {
        hidden_size: get_u32("llama.embedding_length") as usize,
        n_heads: get_u32("llama.attention.head_count") as usize,
        n_kv_heads: get_u32("llama.attention.head_count_kv") as usize,
        head_dim: get_u32("llama.rope.dimension_count") as usize,
        rms_eps: get_f32("llama.attention.layer_norm_rms_epsilon"),
    }
}

// --- RMSNorm ---
// Normalizes a vector by its root-mean-square magnitude, then scales by a
// learned per-element weight. Unlike LayerNorm, there is no mean-subtraction
// step — RMSNorm only rescales magnitude, which is cheaper and works just as
// well in practice for transformers.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(v, w)| v * scale * w).collect()
}

// weight dims are (in, out); computes y = W * x.
fn linear(weight: &[f32], dims: (usize, usize), x: &[f32]) -> Vec<f32> {
    let (in_dim, out_dim) = dims;
    assert_eq!(x.len(), in_dim, "input dim mismatch for linear layer");
    matmul(weight, out_dim, in_dim, x, 1)
}

// --- RoPE (Rotary Position Embeddings) ---
// Encodes token position by rotating pairs of elements within a head's
// vector. Each pair (x[i], x[i+half]) is rotated by an angle depending on
// position and pair index — early pairs rotate fast (high frequency), later
// pairs rotate slowly (low frequency). This lets attention infer relative
// distance between tokens from dot products of rotated vectors, without a
// separate learned position embedding table.
pub fn apply_rope(x: &mut [f32], position: usize, head_dim: usize) {
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

// SiLU(x) = x * sigmoid(x) — the activation used inside SwiGLU.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// --- SwiGLU feed-forward block ---
// Two parallel projections of the same input (gate, up); the gate branch
// passes through SiLU and elementwise-multiplies ("gates") the up branch,
// before a final down-projection back to hidden size. This generally
// outperforms a plain ReLU MLP at the same parameter budget, which is why
// Llama-family models use it.
fn swiglu_mlp(x: &[f32], w: &LayerWeights) -> Vec<f32> {
    let gate = linear(&w.ffn_gate, w.ffn_gate_dims, x);
    let up = linear(&w.ffn_up, w.ffn_up_dims, x);
    let activated: Vec<f32> = gate.iter().zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect();
    linear(&w.ffn_down, w.ffn_down_dims, &activated)
}

// --- Grouped Query Attention (single token, no cache yet) ---
// n_heads query heads share n_kv_heads key/value heads (a group of
// n_heads/n_kv_heads queries reuses the same K/V pair), cutting K/V memory
// bandwidth versus standard multi-head attention. With only one token so
// far, softmax over a single attention score always collapses to 1.0, so
// each head's output is just its value vector — but we still compute the
// scaled dot product to keep this structurally correct for Milestone 3,
// where real multi-token causal attention (and the KV-cache) arrives.
fn attention_single_token(x: &[f32], w: &LayerWeights, cfg: &Config) -> Vec<f32> {
    let q = linear(&w.attn_q, w.attn_q_dims, x); // [n_heads * head_dim]
    let k = linear(&w.attn_k, w.attn_k_dims, x); // [n_kv_heads * head_dim]
    let v = linear(&w.attn_v, w.attn_v_dims, x); // [n_kv_heads * head_dim]

    let group_size = cfg.n_heads / cfg.n_kv_heads;
    let mut out = vec![0f32; cfg.n_heads * cfg.head_dim];

    for h in 0..cfg.n_heads {
        let kv_h = h / group_size; // which shared KV head this query head uses

        let mut qh = q[h * cfg.head_dim..(h + 1) * cfg.head_dim].to_vec();
        let mut kh = k[kv_h * cfg.head_dim..(kv_h + 1) * cfg.head_dim].to_vec();
        let vh = &v[kv_h * cfg.head_dim..(kv_h + 1) * cfg.head_dim];

        apply_rope(&mut qh, 0, cfg.head_dim);
        apply_rope(&mut kh, 0, cfg.head_dim);

        let _score: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum::<f32>()
            / (cfg.head_dim as f32).sqrt();
        // softmax([_score]) == 1.0 always with a single position, so the
        // weighted sum over values is just vh itself.

        out[h * cfg.head_dim..(h + 1) * cfg.head_dim].copy_from_slice(vh);
    }

    linear(&w.attn_output, w.attn_output_dims, &out)
}

// --- Full layer forward pass ---
// x: hidden state for one token, length = hidden_size.
// Order: norm -> attention -> residual add -> norm -> SwiGLU -> residual add.
pub fn forward_layer(x: &[f32], w: &LayerWeights, cfg: &Config) -> Vec<f32> {
    let normed = rms_norm(x, &w.attn_norm, cfg.rms_eps);
    let attn_out = attention_single_token(&normed, w, cfg);
    let residual1: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

    let normed2 = rms_norm(&residual1, &w.ffn_norm, cfg.rms_eps);
    let mlp_out = swiglu_mlp(&normed2, w);
    let residual2: Vec<f32> = residual1.iter().zip(mlp_out.iter()).map(|(a, b)| a + b).collect();

    residual2
}