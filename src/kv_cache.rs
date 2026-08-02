// kv_cache.rs
//
// Storage for cached Key and Value vectors accumulated across generation
// steps. Once a token's Key/Value pair has been computed in a given layer,
// it never changes again, so storing it here lets later steps reuse it
// instead of recomputing it from scratch.
//
// Batched inference needs one independent cache per sequence, since each
// sequence in a batch has its own distinct token history -- weights are
// shared across the batch, but attention state is not.

// One layer's cache for one sequence. For every position processed so far,
// one Key vector and one Value vector per KV head -- not per query head.
// Grouped-query attention means several query heads share the same small
// set of KV heads, which is exactly why this cache stays small: TinyLlama
// has 32 query heads but only 4 KV heads, so the cache is 8x smaller than
// a per-query-head cache would be.
pub struct LayerCache {
    pub keys: Vec<Vec<Vec<f32>>>,   // [position][kv_head] -> head_dim values
    pub values: Vec<Vec<Vec<f32>>>, // [position][kv_head] -> head_dim values
}

impl LayerCache {
    fn new() -> Self {
        LayerCache { keys: Vec::new(), values: Vec::new() }
    }
}

// One sequence's cache across every transformer layer.
pub struct KvCache {
    pub layers: Vec<LayerCache>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        KvCache {
            layers: (0..n_layers).map(|_| LayerCache::new()).collect(),
        }
    }

    // Current memory footprint in bytes: two vectors (Key and Value) per
    // cached position, per layer, each of size n_kv_heads * head_dim floats.
    pub fn memory_bytes(&self, n_kv_heads: usize, head_dim: usize) -> usize {
        let n_positions = self.layers.first().map(|l| l.keys.len()).unwrap_or(0);
        let n_layers = self.layers.len();
        n_positions * n_layers * n_kv_heads * head_dim * 2 * std::mem::size_of::<f32>()
    }
}

// Caches for an entire batch: one KvCache per sequence. Model weights are
// shared across the batch, but each sequence's attention state is separate.
pub struct BatchKvCache {
    pub sequences: Vec<KvCache>,
}

impl BatchKvCache {
    pub fn new(batch_size: usize, n_layers: usize) -> Self {
        BatchKvCache {
            sequences: (0..batch_size).map(|_| KvCache::new(n_layers)).collect(),
        }
    }

    pub fn memory_bytes(&self, n_kv_heads: usize, head_dim: usize) -> usize {
        self.sequences.iter().map(|c| c.memory_bytes(n_kv_heads, head_dim)).sum()
    }
}