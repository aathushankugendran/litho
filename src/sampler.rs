// sampler.rs
//
// Strategies for turning a model's raw output scores (logits) into an
// actual chosen token. argmax (in model.rs) is deterministic and always
// picks the single highest-scoring token -- useful for correctness testing,
// but it makes generated text repetitive in practice. The functions here
// introduce controlled randomness so generation can produce varied,
// natural-sounding text instead of always taking the "safest" word.

use rand::Rng;

// Converts logits into a probability distribution. Subtracting the max
// value before exponentiating avoids overflow (e^large_number blowing up
// to infinity) without changing the resulting probabilities -- softmax is
// shift-invariant.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// Divides every logit by `temperature` before softmax. Below 1.0 sharpens
// the distribution (the model becomes more confident in its top picks);
// above 1.0 flattens it (less-likely tokens become more competitive).
// Exactly 1.0 leaves the distribution unchanged.
fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    logits.iter().map(|l| l / temperature).collect()
}

// Zeroes out every probability except the k highest, then renormalizes so
// the remaining probabilities sum back to 1.0. Guarantees the model can
// never pick something outside its k most likely options, no matter how
// flat the distribution becomes.
fn top_k_filter(probs: &[f32], k: usize) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut filtered = vec![0f32; probs.len()];
    let keep = k.min(indexed.len());
    let kept_sum: f32 = indexed[..keep].iter().map(|(_, p)| p).sum();

    for &(idx, p) in &indexed[..keep] {
        filtered[idx] = p / kept_sum;
    }

    filtered
}

// Keeps adding tokens in order of decreasing probability until their
// cumulative probability crosses `p`, then zeroes out everything else and
// renormalizes. Unlike top-k's fixed count, this adapts the candidate pool
// size to how peaked or flat the distribution is: a very confident
// distribution keeps very few tokens, a flat one keeps many.
fn top_p_filter(probs: &[f32], p: f32) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut filtered = vec![0f32; probs.len()];
    let mut cumulative = 0.0;
    let mut kept: Vec<(usize, f32)> = Vec::new();

    for &(idx, prob) in &indexed {
        if cumulative >= p {
            break;
        }
        cumulative += prob;
        kept.push((idx, prob));
    }

    let kept_sum: f32 = kept.iter().map(|(_, p)| p).sum();
    for &(idx, prob) in &kept {
        filtered[idx] = prob / kept_sum;
    }

    filtered
}

// Draws one token index at random, weighted by the given probability
// distribution -- higher-probability tokens are more likely to be picked,
// but it isn't guaranteed to always be the single highest one.
fn sample_from_distribution(probs: &[f32]) -> usize {
    let mut rng = rand::rng();
    let r: f32 = rng.random();

    let mut cumulative = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return idx;
        }
    }

    // Floating point rounding can leave cumulative just under 1.0; fall
    // back to the last nonzero-probability token rather than panicking.
    probs.iter()
        .enumerate()
        .rev()
        .find(|&(_, &p)| p > 0.0)
        .map(|(idx, _)| idx)
        .expect("probability distribution was empty")
}

pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl SamplingConfig {
    // temperature = 1.0, no filtering -- softmax-weighted random sampling
    // over the full distribution.
    pub fn default_sampling() -> Self {
        SamplingConfig { temperature: 1.0, top_k: None, top_p: None }
    }

    // temperature = 0.0 is a special case handled by the caller (model.rs's
    // argmax), since dividing by zero isn't meaningful here.
    pub fn greedy() -> Self {
        SamplingConfig { temperature: 1.0, top_k: Some(1), top_p: None }
    }
}

// Applies temperature scaling, optional top-k filtering, optional top-p
// filtering (in that order -- each narrows or reshapes the distribution the
// next stage operates on), then draws one token from the result.
pub fn sample(logits: &[f32], config: &SamplingConfig) -> usize {
    let scaled = apply_temperature(logits, config.temperature);
    let mut probs = softmax(&scaled);

    if let Some(k) = config.top_k {
        probs = top_k_filter(&probs, k);
    }
    if let Some(p) = config.top_p {
        probs = top_p_filter(&probs, p);
    }

    sample_from_distribution(&probs)
}