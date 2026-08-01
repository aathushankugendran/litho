import type { ModelInfo } from "../types";

export function ModelInfoBar({ info }: { info: ModelInfo | null }) {
  const fields: [string, string | number][] = info
    ? [
        ["layers", info.n_layers],
        ["hidden", info.hidden_size],
        ["heads", info.n_heads],
        ["kv_heads", info.n_kv_heads],
        ["head_dim", info.head_dim],
        ["vocab", info.vocab_size.toLocaleString()],
      ]
    : [];

  return (
    <div className="mono flex flex-wrap items-center gap-x-6 gap-y-1 border-b border-[var(--border)] bg-[var(--panel)] px-6 py-3 text-xs">
      <span className="text-[var(--amber)] font-semibold tracking-wide">litho</span>
      <span className="text-[var(--text-dim)]">llama-arch inference engine</span>
      <span className="ml-auto flex flex-wrap gap-x-5">
        {fields.map(([label, value]) => (
          <span key={label} className="text-[var(--text-dim)]">
            {label}=<span className="text-[var(--text)]">{value}</span>
          </span>
        ))}
      </span>
    </div>
  );
}