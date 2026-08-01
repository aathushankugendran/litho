# litho

A from-scratch transformer inference engine in Rust — no PyTorch, no ML frameworks.
Implements Llama-style architecture (RoPE, GQA, RMSNorm, SwiGLU, KV-cache) reading
real GGUF checkpoints directly off disk.

## Motivation

This project mirrors the work of an inference runtime team: fused kernels,
KV-cache management, and batching — built from the byte layer up, verified at
every step against trusted references (llama.cpp's own GGUF reader/dequantizer,
and HuggingFace `transformers` for full model behavior).

## Status

- [x] **Milestone 0** — GGUF parser: reads header, metadata, and tensor descriptors from a real GGUF file.
- [x] **Milestone 1** — Dequantization (Q8_0, F32) + naive matmul, verified against llama.cpp's `gguf-py` reference implementation (byte-for-byte match) and hand-computed matmul values.
- [x] **Milestone 2** — Single transformer layer (RMSNorm, RoPE, GQA attention, SwiGLU), verified against HuggingFace `transformers` for one token. See details below.
- [x] **Milestone 3** — Full 22-layer model forward pass, autoregressive generation, and KV-cache. Verified against HuggingFace `transformers` greedy decoding (exact token match), and the KV-cache version verified against this repo's own naive baseline (exact token match). See details below.
- [ ] **Milestone 4** — BPE tokenizer.
- [ ] **Milestone 5** — Sampling (temperature, top-k, top-p).
- [ ] **Milestone 6** — Batching + benchmark harness (tokens/sec at batch 1/4/16, memory).
- [ ] **Milestone 7** — Performance optimization (BLAS/Metal) + comparison vs llama.cpp.

### Milestone 2 verification detail

Layer 0 output for `token_id=1`, first 5 values, compared against HuggingFace
`transformers` running the same architecture at full precision. Small
deviations are attributable to Q8_0 quantization noise (Rust reads a
quantized GGUF checkpoint; Python used full-precision weights):

| Source | Values |
|---|---|
| Rust (this repo) | `[-0.0020121064, -0.010090809, 0.021785222, 0.050538868, -0.038400363]` |
| Python (`transformers`) | `[-0.0019306068, -0.010121642, 0.021579372, 0.050795078, -0.038452946]` |

### Milestone 3 verification detail

**Naive multi-token generation vs. HuggingFace `transformers`** — same
prompt token IDs (`[1, 2, 3]`), 3 new tokens, greedy decoding on both sides.
Exact match:

| Source | Values |
|---|---|
| Rust (this repo) | `[1, 2, 3, 29966, 29989, 1792]` |
| Python (`transformers`) | `[1, 2, 3, 29966, 29989, 1792]` |

**KV-cache vs. naive baseline** — since the KV-cache changes only how much
work is repeated (never the underlying math), it must produce identical
output to the already-verified naive version given the same inputs. Checked
programmatically via `assert_eq!` on the full generated token sequence; both
approaches agree exactly.

**Why KV-cache?** Naive generation recomputes Key/Value vectors for every
prior token on every generation step. The KV-cache stores them once and
reuses them, so only the newest token's K/V need to be computed each step.
Measured on the same prompt and hardware, generating an increasing number of
new tokens:

| New tokens | Naive (ms) | Cached (ms) | Speedup |
|---|---|---|---|
| 5  | 14,544  | 4,934  | 2.95x  |
| 15 | 88,807  | 11,386 | 7.80x  |
| 30 | 316,292 | 24,115 | 13.12x |

The speedup grows with sequence length, not just the absolute time saved:
naive attention cost grows roughly quadratically with the number of tokens
generated (it re-does more accumulated work at every step), while the
KV-cache keeps this close to linear. This is exactly the tradeoff a
KV-cache is meant to make, and it becomes more valuable the longer
generation runs.

(These are single-run wall-clock measurements on one machine, meant to
illustrate the trend — Milestone 6 adds a proper benchmarking harness with
repeated trials.)

## Model used for development

[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF),
Q8_0 quantization.

## Running

```bash
cargo run --release
```

(Model path is currently hardcoded in `main.rs`; will become a CLI argument in
Milestone 4, once a real tokenizer replaces the placeholder token IDs.)

## Verification approach

Every numerical component is checked against an independent source before
moving on:

- **Dequantization** — cross-checked against llama.cpp's own `gguf-py`
  library (byte-for-byte match on real tensor data).
- **Matmul** — checked against hand-computed values.
- **Single transformer layer** (RMSNorm, RoPE, GQA, SwiGLU) — cross-checked
  against HuggingFace `transformers` running the same model at full
  precision.
- **Full model + autoregressive generation** — cross-checked against
  HuggingFace `transformers` greedy decoding (exact token match).
- **KV-cache** — cross-checked against this repo's own verified naive
  baseline (exact token match), since caching must not change model
  behavior, only reduce redundant computation.

This project prioritizes correctness at each layer over speed until
Milestone 7.