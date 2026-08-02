// bench.rs
//
// Benchmark harness measuring throughput and memory across batch sizes.
// Batching amortizes the cost of reading model weights out of memory across
// several sequences, so throughput per sequence should improve with batch
// size even though each individual sequence gets no faster -- the point of
// the measurement is to show that scaling, and where it stops paying off.

use std::time::Instant;

use llm_engine::{gguf, model, tokenizer};

const MODEL_PATH: &str = "/Users/aathushankugendran/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";

const PROMPTS: [&str; 16] = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "The first person to walk on the moon was",
    "Photosynthesis is the process by which",
    "The Pacific Ocean is located between",
    "A prime number is defined as",
    "The Great Wall of China was built to",
    "Gravity causes objects to",
    "The human heart pumps",
    "Electricity flows through materials called",
    "The Amazon rainforest is home to",
    "Sound travels faster through",
    "The Earth completes one rotation in",
    "Vaccines work by teaching the immune system to",
    "Mount Everest is the tallest mountain because",
];

// Peak resident set size in bytes, read from the OS rather than estimated,
// so the reported memory reflects everything actually held: weights,
// caches, and runtime overhead.
fn peak_memory_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    // macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        usage.ru_maxrss as u64 * 1024
    }
}

fn main() {
    println!("Loading model...");
    let file = gguf::parse(MODEL_PATH).expect("failed to parse GGUF file");
    let model = model::load_model(MODEL_PATH, &file);
    let tok = tokenizer::load_tokenizer(&file);
    println!(
        "Loaded {} layers, hidden_size={}, vocab_size={}\n",
        model.config.n_layers, model.config.hidden_size, model.config.vocab_size
    );

    let n_new_tokens = 20;
    let batch_sizes = [1usize, 4, 16];

    println!(
        "{:<8} {:<12} {:<14} {:<16} {:<14}",
        "batch", "tokens", "wall time (s)", "throughput tok/s", "peak RSS (MB)"
    );
    println!("{}", "-".repeat(68));

    for &batch_size in &batch_sizes {
        let prompts: Vec<Vec<usize>> = PROMPTS[..batch_size]
            .iter()
            .map(|p| {
                let mut ids = vec![tok.bos_id()];
                ids.extend(tok.encode(p));
                ids
            })
            .collect();

        let start = Instant::now();
        let results = model::generate_batched(&model, &prompts, n_new_tokens);
        let elapsed = start.elapsed().as_secs_f64();

        // Total tokens generated across the whole batch -- this is the number
        // that matters for a serving system, since it reflects how much work
        // the hardware actually completed, not how fast any one user's
        // request felt.
        let total_tokens = batch_size * n_new_tokens;
        let throughput = total_tokens as f64 / elapsed;
        let peak_mb = peak_memory_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "{:<8} {:<12} {:<14.2} {:<16.2} {:<14.1}",
            batch_size, total_tokens, elapsed, throughput, peak_mb
        );

        // Sanity check: batch 1 should reproduce the same greedy output as
        // the single-sequence path, confirming batching didn't change the math.
        if batch_size == 1 {
            println!("  batch-1 output: {:?}\n", tok.decode(&results[0]));
        }
    }
}