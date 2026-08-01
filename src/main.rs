mod gguf;
mod tensor;
mod model;
mod kv_cache;
mod tokenizer;
mod sampler;

use std::time::Instant;

fn main() {
    let path = "/Users/aathushankugendran/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    let file = gguf::parse(path).expect("failed to parse GGUF file");

    println!("Loading model...");
    let model = model::load_model(path, &file);
    println!(
        "Loaded {} layers, hidden_size={}, vocab_size={}",
        model.config.n_layers, model.config.hidden_size, model.config.vocab_size
    );

    let prompt_ids = vec![1usize, 2, 3];

    // generate_cached is the real generation path going forward -- naive
    // generation is kept only as a correctness baseline (see benchmark
    // below and README) and is not used for actual output.
    let generated = model::generate_cached(&model, &prompt_ids, 10);
    println!("\ngenerated token ids: {:?}", generated);

    // --- Naive vs. cached benchmark (see README "Why KV-cache?" section) ---
    let naive_check = model::generate_naive(&model, &prompt_ids, 5);
    let cached_check = model::generate_cached(&model, &prompt_ids, 5);
    assert_eq!(naive_check, cached_check, "cache diverged from naive -- fix before trusting either");

    println!("\n{:<12} {:<15} {:<15} {:<10}", "new_tokens", "naive (ms)", "cached (ms)", "speedup");
    for &n in &[5, 15, 30] {
        let start = Instant::now();
        model::generate_naive(&model, &prompt_ids, n);
        let naive_ms = start.elapsed().as_millis();

        let start = Instant::now();
        model::generate_cached(&model, &prompt_ids, n);
        let cached_ms = start.elapsed().as_millis();

        let speedup = naive_ms as f64 / cached_ms.max(1) as f64;
        println!("{:<12} {:<15} {:<15} {:<10.2}x", n, naive_ms, cached_ms, speedup);
    }
}