export interface ModelInfo {
  n_layers: number;
  hidden_size: number;
  n_heads: number;
  n_kv_heads: number;
  head_dim: number;
  vocab_size: number;
}

export interface StreamEvent {
  step: number;
  token_id: number;
  piece: string;
  elapsed_ms: number;
  cache_kb: number;
}

export type Mode = "naive" | "cached";

export interface SamplingConfig {
  temperature: number;
  top_k: number | null;
  top_p: number | null;
}