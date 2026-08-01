mod gguf;
mod tensor;
mod model;
mod kv_cache;
mod tokenizer;
mod sampler;

fn main() {
    let path = "/Users/aathushankugendran/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    let file = gguf::parse(path).expect("failed to parse GGUF file");

    println!("Loading model...");
    let model = model::load_model(path, &file);
    println!(
        "Loaded {} layers, hidden_size={}, vocab_size={}",
        model.config.n_layers, model.config.hidden_size, model.config.vocab_size
    );

    // No tokenizer yet (Milestone 4), so we pick arbitrary token ids as a
    // stand-in prompt purely to exercise the generation loop end to end.
    let prompt_ids = vec![1usize, 2, 3];
    let n_new_tokens = 10;

    println!("\n--- naive generation ---");
    let naive_result = model::generate_naive(&model, &prompt_ids, n_new_tokens);
    println!("naive:  {:?}", naive_result);

    println!("\n--- cached generation ---");
    let cached_result = model::generate_cached(&model, &prompt_ids, n_new_tokens);
    println!("cached: {:?}", cached_result);

    assert_eq!(
        naive_result, cached_result,
        "KV-cache output diverged from naive output -- these must match exactly"
    );
    println!("\nnaive and cached generation produced identical token sequences.");
}