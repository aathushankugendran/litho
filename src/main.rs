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
}