# litho

A from-scratch transformer inference engine, written entirely in Rust — no PyTorch, no ML frameworks, no external inference libraries. It reads a real GGUF model checkpoint directly off disk, implements every piece of Llama-style architecture (RoPE, GQA, RMSNorm, SwiGLU, KV-cache) by hand, and serves it through a live HTTP API with a browser dashboard on top.

Every numerical component in this project is independently verified against a trusted reference before being built upon — not just "it compiles and looks right."

**Live dashboard:** [litho](https://main.d2jztxcsysdow0.amplifyapp.com/) (React/TypeScript, streams generation in real time)

## About the name

*Litho* comes from the Greek *lithos*, meaning stone — the bedrock something is built on. This project is the ground floor: no PyTorch, no inference libraries, no framework standing between the code and the math. Every layer of the transformer, every byte of the GGUF file, every optimization pass was written by hand, so the foundation itself is fully understood rather than borrowed.

---

## Table of contents

- [Motivation](#motivation)
- [Architecture overview](#architecture-overview)
- [Milestone 0 — GGUF parser](#milestone-0--gguf-parser)
- [Milestone 1 — Dequantization + matmul](#milestone-1--dequantization--matmul)
- [Milestone 2 — Single transformer layer](#milestone-2--single-transformer-layer)
- [Milestone 3 — Full model, KV-cache, generation](#milestone-3--full-model-kv-cache-generation)
- [Milestone 4 — Tokenizer](#milestone-4--tokenizer)
- [Milestone 5 — Sampling](#milestone-5--sampling)
- [Milestone 6 — Batching](#milestone-6--batching)
- [Milestone 7 — Performance optimization](#milestone-7--performance-optimization)
- [Running the project](#running-the-project)
- [Known limitations and future work](#known-limitations-and-future-work)

---

## Motivation

This project mirrors the work of an inference runtime team: fused kernels, KV-cache management, quantization, and batching — built from the byte layer up, and measured rigorously at every stage rather than assumed to work.

**Model used throughout:** [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF), Q8_0 quantization.

---

## Architecture overview

```text
GGUF file on disk
│
▼
GGUF parser (header, metadata, tensor index)
│
▼
Dequantization (Q8_0 → f32)
│
▼
22× transformer layers
(RMSNorm → GQA attention with RoPE → SwiGLU MLP)
│
▼
KV-cache
(per-sequence, reused across generation steps)
│
▼
Sampling
(greedy / temperature / top-k / top-p)
│
▼
BPE tokenizer
(encode/decode)
│
▼
Axum HTTP server
(SSE streaming)
│
▼
React dashboard
```

---

## Milestone 0 — GGUF parser

**What it does:** reads the GGUF binary format's fixed header, all metadata key/value pairs—including the array/tagged-union type system—and every tensor's name, shape, and byte offset, without touching any of the actual weight data yet.

**Result on the TinyLlama checkpoint used throughout this project:**

| Field | Value |
|---|---:|
| GGUF version | 3 |
| Metadata entries | 23 |
| Tensor count | 201 |
| Architecture | `llama` |
| Layers | 22 |
| Hidden size | 2048 |
| Attention heads | 32 |
| KV heads | 4 |
| Head dimension | 64 |

**Verification:** the raw header bytes were manually decoded from a hex dump—including the magic number, version, tensor count, and metadata count—and cross-checked against the parser's output before any code was trusted.

---

## Milestone 1 — Dequantization + matmul

**What it does:** converts compressed on-disk weights back into usable `f32` values and implements matrix multiplication, the operation that nearly every part of a transformer reduces to.

Q8_0 stores one shared 16-bit floating-point scale for each block of 32 signed 8-bit integers.

### Dequantization verification

Compared against llama.cpp's reference implementation, `gguf-py`, using the first five values of `blk.0.attn_q.weight`:

| Source | Values |
|---|---|
| This engine (Rust) | `[-0.0014365911, -0.0024311543, -0.0074039698, -0.01403439, -0.0028731823]` |
| llama.cpp (`gguf-py`, Python) | `[-0.00143659, -0.00243115, -0.00740397, -0.01403439, -0.00287318]` |

The results match to the precision printed.

### Matmul verification

Compared against hand-computed arithmetic:

```text
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

A × B = [[19, 22], [43, 50]]
```

The engine returned:

```text
[19, 22, 43, 50]
```

This matches the hand calculation exactly.

---

## Milestone 2 — Single transformer layer

**What it does:** implements one full Llama-style transformer block applied to a single token:

- RMSNorm
- Grouped-query attention
- Rotary positional embeddings
- Residual connections
- SwiGLU feed-forward network

### Verification against Hugging Face

The layer-0 output for `token_id=1` was compared against Hugging Face `transformers` running the same architecture at full, unquantized precision.

| Source | First five values |
|---|---|
| This engine (Rust, Q8_0) | `[-0.0020121064, -0.010090809, 0.021785222, 0.050538868, -0.038400363]` |
| `transformers` (Python, full precision) | `[-0.0019306068, -0.010121642, 0.021579372, 0.050795078, -0.038452946]` |

The outputs match in sign and magnitude to roughly two or three significant figures.

The small remaining difference is expected because this engine reads Q8_0-quantized weights, while the Python reference uses full-precision `f32` weights.

---

## Milestone 3 — Full model, KV-cache, generation

**What it does:** stacks all 22 transformer layers, adds causal multi-token attention, and implements autoregressive generation in two ways:

- **Naive generation:** recomputes every earlier token's attention Key and Value vectors on every generation step.
- **KV-cached generation:** computes each token's Key and Value vectors once and reuses them during later steps.

### Generation verification

Naive greedy generation was compared against Hugging Face `transformers` using the same prompt token IDs and generating three new tokens.

| Source | Token IDs |
|---|---|
| This engine (Rust) | `[1, 2, 3, 29966, 29989, 1792]` |
| `transformers` (Python) | `[1, 2, 3, 29966, 29989, 1792]` |

The generated sequences matched exactly.

### KV-cache verification

Caching changes only how much work is repeated. It should not change the underlying mathematical result.

The full output sequence from cached generation was compared against the naive implementation using Rust's `assert_eq!` macro.

The generated token sequences were identical in every test run.

### KV-cache performance

Measured using the same prompt and hardware:

| New tokens generated | Naive (ms) | Cached (ms) | Speedup |
|---:|---:|---:|---:|
| 5 | 14,544 | 4,934 | 2.95× |
| 15 | 88,807 | 11,386 | 7.80× |
| 30 | 316,292 | 24,115 | 13.12× |

The speedup increases with sequence length.

Naive generation repeatedly recomputes accumulated attention work, causing its cost to grow roughly quadratically. The KV-cache avoids this repeated work and keeps generation cost closer to linear.

### KV-cache memory usage

Measured memory cost:

```text
Approximately 44 KB per cached token position
```

This covers all 22 transformer layers and matches the hand-derived formula:

```text
2 × n_layers × n_kv_heads × head_dim × bytes_per_value
```

The leading `2` accounts for both Key and Value vectors.

Grouped-query attention uses four KV heads instead of 32 query heads, making the cache approximately eight times smaller than a cache that stored separate values for every query head.

---

## Milestone 4 — Tokenizer

**What it does:** implements SentencePiece-style Byte-Pair Encoding, the tokenization scheme used by Llama models.

The tokenizer reads the vocabulary and merge rules directly from the GGUF file's metadata and includes byte-fallback handling for characters outside the trained vocabulary.

### Round-trip verification

| Step | Value |
|---|---|
| Input | `"The capital of France is"` |
| Encoded token IDs | `[1, 450, 7483, 310, 3444, 338]` |
| Decoded output | `"The capital of France is"` |

The decoded text matches the original input exactly.

Token ID `1` is the beginning-of-sequence special token and is correctly excluded from the human-readable decoded output.

### First end-to-end text generation

This was the first point in the project where raw text went in and coherent text came out using only code from this repository.

> **Prompt:** The capital of France is  
> **Generated:** The capital of France is Paris, which is the capital of France.

No external tokenizer or model library was used anywhere in the generation pipeline.

---

## Milestone 5 — Sampling

**What it does:** adds several decoding strategies on top of deterministic greedy generation:

- Temperature scaling
- Top-k filtering
- Top-p nucleus filtering

These methods allow generation to produce varied output instead of always selecting the single highest-scoring token.

### Greedy versus sampled generation

Sampling configuration:

```text
temperature = 0.8
top_k = 40
top_p = 0.9
```

| Mode | Output |
|---|---|
| Greedy | `"The capital of France is Paris, which is the capital of France."` |
| Sampled | `"The capital of France is Paris and is the capital city, and it'"` |

Both outputs correctly identify Paris, while the sampled version diverges in phrasing because controlled randomness has been introduced.

### Observed model limitation

With greedy decoding on vague or open-ended prompts, TinyLlama can fall into repetition loops, such as:

```text
I'm not like I'm not like...
```

TinyLlama is a relatively small 1.1-billion-parameter model. Greedy decoding has no mechanism to escape a locally high-probability repeating pattern.

Enabling sampling reliably helps break these loops. This is a limitation of the model and decoding strategy rather than a bug in the inference engine.

---

## Milestone 6 — Batching

**What it does:** processes multiple prompts simultaneously by restructuring every transformer layer to operate on `[batch × hidden]` matrices instead of individual vectors.

Each sequence maintains its own independent KV-cache.

### Why batching should help

Generating a token requires reading the model's approximately 1.1 billion parameters from memory.

With one sequence, each weight is read and used once. The processor spends much of its time waiting for memory rather than performing arithmetic.

With batching, the same weights can theoretically be read once and reused across several sequences. This amortizes memory access across more useful computation.

### Benchmark setup

The benchmark harness measures:

- Total throughput across the batch
- Wall-clock execution time
- Peak resident memory through the operating system's `getrusage`
- Batch sizes of 1, 4, and 16
- 20 generated tokens per sequence

### Results before kernel optimization

| Batch | Total tokens | Wall time (s) | Throughput (tok/s) | Peak RSS (MB) |
|---:|---:|---:|---:|---:|
| 1 | 20 | 16.31 | 1.23 | 4,278.8 |
| 4 | 80 | 63.98 | 1.25 | 4,284.1 |
| 16 | 320 | 266.87 | 1.20 | 4,291.5 |

Throughput remained nearly flat across all batch sizes.

Wall time scaled almost perfectly linearly with batch size:

```text
16.31 seconds → 63.98 seconds → 266.87 seconds
```

This showed that the engine was performing `batch_size` times more work without efficiently sharing computation between sequences.

Batching by itself provided no benefit because the underlying matrix multiplication kernel could not exploit the batch structure.

This finding directly motivated Milestone 7.

---

## Milestone 7 — Performance optimization

Performance optimization was completed in three stages. Each stage used the same benchmark harness, allowing direct before-and-after comparison.

### Stage 1 — Cache-aware matmul

#### Problem

The original matrix multiplication used the loop order:

```text
i → j → p
```

This caused the innermost loop to access the second matrix using a memory stride of `n`, where `n` is the batch size.

Each memory fetch loaded an entire CPU cache line but used only one value from it. The first matrix's row was also reread for every batch column.

#### Fix

The loop order was changed to:

```text
i → p → j
```

This allows the innermost loop to access memory sequentially and use every value loaded into a cache line.

It also allows a value from the first matrix to be loaded into a register and reused across every batch column.

This creates the intended batching behavior:

```text
Read each weight once and use it n times.
```

#### Results

| Batch | Throughput (tok/s) | Improvement over naive matmul |
|---:|---:|---:|
| 1 | 1.04 | Slightly slower |
| 4 | 2.70 | 2.16× |
| 16 | 3.32 | 2.77× |

Batching began scaling with batch size, confirming that the bottleneck was the memory access pattern rather than the concept of batching itself.

Batch size 1 was slightly slower because the additional loop and branching overhead could not be amortized across multiple sequences.

---

### Stage 2 — Accelerate / BLAS

#### Change

The hand-written matrix multiplication implementation was replaced with Apple's Accelerate framework using:

```text
cblas_sgemm
```

Accelerate provides a professionally optimized BLAS implementation with:

- SIMD vector instructions
- Multilevel cache blocking
- Apple Silicon-specific optimizations
- Highly optimized matrix multiplication kernels

This is the same general category of optimized library that llama.cpp can use.

#### Results

| Batch | Naive matmul (tok/s) | Reordered loop (tok/s) | BLAS (tok/s) |
|---:|---:|---:|---:|
| 1 | 1.23 | 1.04 | **9.37** |
| 4 | 1.25 | 2.70 | **26.91** |
| 16 | 1.20 | 3.32 | **74.08** |

Batch size 16 improved by approximately **61×** compared with the unoptimized starting point:

```text
74.08 ÷ 1.20 ≈ 61.7×
```

This improvement came entirely from software-level optimization on the same hardware.

Batching now scaled close to linearly:

```text
9.37 tok/s → 26.91 tok/s → 74.08 tok/s
```

The remaining sub-linear scaling is likely caused by attention computation, which still runs as unbatched scalar code for each sequence.

That is now the next identified performance bottleneck.

#### Correctness verification

The output for the same greedy prompt remained byte-identical across all three matrix multiplication implementations:

- Naive matmul
- Cache-aware reordered matmul
- Accelerate BLAS matmul

Example output:

```text
The capital of France is Paris, which is the capital of France.
```

Each optimization changed execution speed without changing the model's result.

---

### Stage 3 — Comparison against llama.cpp

The engine was compared against llama.cpp using:

- The same GGUF model
- The same prompt
- The same machine
- Greedy decoding
- llama.cpp's `llama-cli` tool

| Engine | Batch-1 throughput |
|---|---:|
| This engine — naive matmul | 1.23 tok/s |
| This engine — reordered loop | 1.04 tok/s |
| This engine — BLAS | 9.37 tok/s |
| **llama.cpp** | **110.2 tok/s** |

llama.cpp remains approximately 11.7 times faster than this engine's BLAS implementation:

```text
110.2 ÷ 9.37 ≈ 11.7×
```

#### Why llama.cpp is faster

1. **Quantized-native kernels**

   llama.cpp multiplies directly against Q8_0 integer weights using specialized kernels.

   This engine currently dequantizes weights to `f32` before calling BLAS, which introduces additional memory traffic and loses the bandwidth advantage of the smaller quantized representation.

2. **Architecture-specific SIMD**

   llama.cpp uses hand-written SIMD kernels designed for specific CPU architectures.

   This project currently relies on a generic BLAS call.

3. **Engineering maturity**

   llama.cpp has received continuous community-wide performance optimization since 2023.

   This project's initial optimization pass was completed over a much shorter development period.

The performance gap is reported rather than hidden.

Understanding why a mature inference runtime outperforms a from-scratch implementation—and identifying the techniques responsible—is a more valuable engineering result than presenting an unrealistic or unexplained benchmark.

---

## Running the project

### Backend

Run the Rust inference engine and HTTP server:

```bash
cargo run --release --bin server
```

The server runs on:

```text
http://localhost:3000
```

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/model-info` | Returns loaded model information |
| `POST` | `/generate` | Generates text through Server-Sent Events |

### CLI demo

Run the inference engine without the HTTP server:

```bash
cargo run --release --bin llm-engine
```

### Benchmark harness

Run the performance benchmarks:

```bash
cargo run --release --bin bench
```

### Frontend dashboard

Clone or open the [`litho-ui`](https://github.com/aathushankugendran/litho-ui) repository, then run:

```bash
cd litho-ui
npm install
npm run dev
```

The backend server must be running simultaneously on:

```text
http://localhost:3000
```

---

## Known limitations and future work

- **Hardcoded model path:** the model path is currently hardcoded in `main.rs` and `server.rs` instead of being accepted through a command-line argument.
- **Scalar batched attention:** attention is still processed independently for each sequence and is now the next identified performance bottleneck.
- **Limited quantization support:** only Q8_0 and F32 dequantization are implemented. Loading Q4_K_M or other unsupported formats will panic.
- **No continuous batching:** all sequences in a batch generate to the same fixed length instead of replacing completed sequences with new requests.
- **CPU-only inference:** there is currently no GPU or Metal backend.
- **No distributed inference:** multi-node and distributed model execution are not implemented.
