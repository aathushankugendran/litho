mod gguf;
mod tensor;
mod model;
mod kv_cache;
mod tokenizer;
mod sampler;

fn main() {
    let path = "/Users/aathushankugendran/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    let file = gguf::parse(path).expect("failed to parse GGUF file");

    println!("GGUF version: {}", file.version);
    println!("Metadata entries: {}", file.metadata.len());
    println!("Tensor count: {}", file.tensors.len());

    println!("\n--- metadata (first 10) ---");
    for (key, value) in file.metadata.iter().take(10) {
        println!("{key}: {value:?}");
    }

    println!("\n--- tensors (first 10) ---");
    for t in file.tensors.iter().take(10) {
        println!("{} | dims: {:?} | dtype: {}", t.name, t.dims, t.dtype);
    }

    // Sanity-check dequantization: a small F32 tensor first (norm weights
    // are stored unquantized), then a real Q8_0 tensor.
    let norm_tensor = file.tensors.iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .expect("tensor not found");

    let values = tensor::load_tensor(path, file.tensor_data_offset, norm_tensor);
    println!("\n--- blk.0.attn_norm.weight (first 5 values) ---");
    println!("{:?}", &values[..5]);

    let q_tensor = file.tensors.iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("tensor not found");

    let q_values = tensor::load_tensor(path, file.tensor_data_offset, q_tensor);
    println!("\n--- blk.0.attn_q.weight (first 5 dequantized values) ---");
    println!("{:?}", &q_values[..5]);

    // Sanity-check matmul with hand-checkable numbers.
    // A = [[1, 2], [3, 4]]  (2x2)
    // B = [[5, 6], [7, 8]]  (2x2)
    // Expected: [[19, 22], [43, 50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = tensor::matmul(&a, 2, 2, &b, 2);
    println!("\n--- matmul sanity check ---");
    println!("{:?} (expected [19, 22, 43, 50])", result);

    // --- Milestone 2: run one real transformer layer on one real token ---
    let cfg = model::load_config(&file);
    let layer0 = model::load_layer(path, &file, 0);

    // Grab this token's embedding row out of the embedding table.
    // token_embd.weight dims are [hidden_size, vocab_size]; row for token_id
    // is a contiguous slice of length hidden_size.
    let embd_tensor = file.tensors.iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("embedding tensor not found");
    let embd_table = tensor::load_tensor(path, file.tensor_data_offset, embd_tensor);

    let token_id = 1; // arbitrary token for now; real tokenizer comes in Milestone 4
    let hidden_size = cfg.hidden_size;
    let x = embd_table[token_id * hidden_size..(token_id + 1) * hidden_size].to_vec();

    let output = model::forward_layer(&x, &layer0, &cfg);
    println!("\n--- layer 0 output (first 5 values, token_id={token_id}) ---");
    println!("{:?}", &output[..5]);
}