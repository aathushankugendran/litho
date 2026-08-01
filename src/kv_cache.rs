// kv_cache.rs
//
// Storage for cached Key and Value vectors accumulated across generation
// steps. Once a token's Key/Value pair has been computed in a given layer,
// it never changes again, so storing it here lets later steps reuse it
// instead of recomputing it from scratch. See model.rs for the attention
// function that actually reads from and writes to this cache.

// One layer's cache. For every position processed so far, one Key vector
// and one Value vector per KV head -- not per query head. Grouped-query
// attention means several query heads share the same small set of KV
// heads, which is exactly why this cache stays small: TinyLlama has 32
// query heads but only 4 KV heads, so the cache is 8x smaller than a naive
// per-query-head cache would be.
pub struct LayerCache {
    pub keys: Vec<Vec<Vec<f32>>>,   // indexed [position][kv_head] -> head_dim values
    pub values: Vec<Vec<Vec<f32>>>, // indexed [position][kv_head] -> head_dim values
}

impl LayerCache {
    fn new() -> Self {
        LayerCache { keys: Vec::new(), values: Vec::new() }
    }
}

// The full model's cache: one LayerCache per transformer layer, since each
// layer computes its own independent set of Key/Value vectors.
pub struct KvCache {
    pub layers: Vec<LayerCache>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        KvCache {
            layers: (0..n_layers).map(|_| LayerCache::new()).collect(),
        }
    }

    // Computes the cache's current memory footprint in bytes: two vectors
    // (Key and Value) per cached position, per layer, each of size
    // n_kv_heads * head_dim floats. This is the same formula worked out by
    // hand earlier -- surfacing it here lets the frontend show it update
    // live as generation proceeds.
    pub fn memory_bytes(&self, n_kv_heads: usize, head_dim: usize) -> usize {
        let n_positions = self.layers.first().map(|l| l.keys.len()).unwrap_or(0);
        let n_layers = self.layers.len();
        n_positions * n_layers * n_kv_heads * head_dim * 2 * std::mem::size_of::<f32>()
    }
}