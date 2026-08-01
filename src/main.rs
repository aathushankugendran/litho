mod gguf;
mod tensor;
mod model;
mod kv_cache;
mod tokenizer;
mod sampler;

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
    println!("encoded token ids: {:?}", prompt_ids);

    let round_trip = tok.decode(&prompt_ids);
    println!("decoded back: {round_trip:?}");

    let generated = model::generate_cached(&model, &prompt_ids, 10);
    println!("\ngenerated token ids: {:?}", generated);
    println!("generated text: {:?}", tok.decode(&generated));

    let sampling = SamplingConfig { temperature: 0.8, top_k: Some(40), top_p: Some(0.9) };
    let sampled = model::generate_cached_sampled(&model, &prompt_ids, 10, &sampling);
    println!("\nsampled generation (temp=0.8, top_k=40, top_p=0.9): {:?}", sampled);
    println!("sampled text: {:?}", tok.decode(&sampled));
}