// lib.rs
//
// Exposes the engine's modules as a library so multiple binaries (the CLI
// demo in main.rs, and the HTTP server in bin/server.rs) can share the same
// GGUF parsing, tensor math, model, cache, tokenizer, and sampler code
// without duplicating it.

pub mod gguf;
pub mod tensor;
pub mod model;
pub mod kv_cache;
pub mod tokenizer;
pub mod sampler;