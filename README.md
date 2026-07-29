# litho

A from-scratch transformer inference engine in Rust — no PyTorch, no ML frameworks.
Implements Llama-style architecture (RoPE, GQA, RMSNorm, SwiGLU, KV-cache) reading
real GGUF checkpoints directly off disk.

## Motivation

This project mirrors the work of an inference runtime team: fused kernels,
KV-cache management, and batching — built from the byte layer up, verified at
every step against trusted references (llama.cpp's own GGUF reader/dequantizer).

## Status

- [x] **Milestone 0** — GGUF parser: reads header, metadata, and tensor descriptors
      from a real GGUF file.
- [x] **Milestone 1** — Dequantization (Q8_0, F32) + naive matmul, verified against
      llama.cpp's `gguf-py` reference implementation (byte-for-byte match) and
      hand-computed matmul values.
- [ ] **Milestone 2** — Single transformer layer (RMSNorm, RoPE, GQA attention, SwiGLU),
      verified against a Python reference for one token.
- [ ] **Milestone 3** — Full model forward pass + KV-cache + autoregressive generation.
- [ ] **Milestone 4** — BPE tokenizer.
- [ ] **Milestone 5** — Sampling (temperature, top-k, top-p).
- [ ] **Milestone 6** — Batching + benchmark harness (tokens/sec at batch 1/4/16, memory).
- [ ] **Milestone 7** — Performance optimization (BLAS/Metal) + comparison vs llama.cpp.

## Model used for development

[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF),
Q8_0 quantization.

## Running

```bash
cargo run
```

(Model path is currently hardcoded in `main.rs`; will become a CLI argument in
Milestone 3.)

## Verification approach

Every numerical component is checked against an independent source before moving
on — dequantization was cross-checked against llama.cpp's own `gguf-py` library,
and matmul was checked against hand-computed values. This project prioritizes
correctness at each layer over speed until Milestone 7.
