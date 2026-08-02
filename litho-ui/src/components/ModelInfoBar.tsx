import { useEffect, useRef, useState } from "react";
import type { ModelInfo } from "../types";

// Short plain-English notes on the specs, so someone unfamiliar with
// transformer internals can read the header without prior context.
const SPEC_NOTES: Record<string, string> = {
  layers: "stacked transformer blocks the text passes through",
  hidden: "size of the vector representing each token",
  heads: "parallel attention heads per layer",
  kv_heads: "shared key/value heads — fewer than query heads, which cuts memory use",
  head_dim: "numbers per attention head",
  vocab: "distinct tokens the model can produce",
};

export function ModelInfoBar({ info }: { info: ModelInfo | null }) {
  const specs: [string, string | number][] = info
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
    <header className="border-b border-[var(--border)] bg-[var(--panel)] px-6 py-4">
      <div className="mx-auto max-w-6xl">
        <div className="mono flex flex-wrap items-center gap-x-3">
          <span className="text-base font-semibold tracking-wide text-[var(--amber)]">litho</span>
          {info && (
            <>
              <span className="text-sm text-[var(--text-dim)]">running</span>
              <span className="text-sm text-[var(--text)]">{info.name}</span>
              <span className="flex items-center gap-1.5">
                <span className="rounded bg-[var(--bg)] px-1.5 py-0.5 text-xs text-[var(--cyan)]">
                  {info.quantization}
                </span>
                <QuantizationHelp quantization={info.quantization} />
              </span>
            </>
          )}
        </div>

        <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-[var(--text-dim)]">
          A transformer inference engine written from scratch in Rust — no PyTorch, no ML libraries.
          It reads a real model checkpoint off disk, runs the math itself, and streams the result
          back token by token so you can watch what the engine is actually doing.
        </p>

        <div className="mono mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs">
          {specs.map(([label, value]) => (
            <span key={label} className="text-[var(--text-dim)]" title={SPEC_NOTES[label]}>
              {label}=<span className="text-[var(--text)]">{value}</span>
            </span>
          ))}
        </div>
      </div>
    </header>
  );
}

// A click-to-open explainer for the quantization format. Kept as a popover
// rather than a tooltip because the content is a few sentences plus a link,
// which is too much for a hover target.
function QuantizationHelp({ quantization }: { quantization: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click or Escape, so the popover behaves the way people
  // expect without trapping them in it.
  useEffect(() => {
    if (!open) return;

    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };

    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label={`What does ${quantization} mean?`}
        aria-expanded={open}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-[var(--text-dim)] text-[10px] leading-none text-[var(--text-dim)] transition-colors hover:border-[var(--cyan)] hover:text-[var(--cyan)]"
      >
        ?
      </button>

      {open && (
        <div className="absolute left-0 top-6 z-20 w-80 rounded border border-[var(--border)] bg-[var(--panel)] p-4 shadow-xl">
          <div className="mono mb-2 text-xs text-[var(--cyan)]">{quantization} quantization</div>

          <p className="mb-3 text-xs leading-relaxed text-[var(--text-dim)]">
            The model's weights are stored as 8-bit integers instead of 32-bit decimals, with one
            shared scaling factor per block of 32 values. That cuts the file from roughly 4.4GB to
            1.1GB with very little loss in output quality.
          </p>

          <p className="mb-3 text-xs leading-relaxed text-[var(--text-dim)]">
            Q8_0 was chosen here because it's the simplest real quantization format to implement
            correctly — one scale, one integer per weight, no nested blocks. That made it possible
            to verify this engine's dequantization byte-for-byte against llama.cpp's own reference
            implementation before building anything on top of it.
          </p>

          <div className="flex flex-col gap-1.5">
            <a
              href="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
              target="_blank"
              rel="noreferrer"
              className="mono text-xs text-[var(--amber)] underline decoration-dotted underline-offset-4 hover:text-[var(--text)]"
            >
              the model on Hugging Face →
            </a>
            <a
              href="https://github.com/aathushankugendran/litho"
              target="_blank"
              rel="noreferrer"
              className="mono text-xs text-[var(--amber)] underline decoration-dotted underline-offset-4 hover:text-[var(--text)]"
            >
              how dequantization was verified →
            </a>
          </div>
        </div>
      )}
    </div>
  );
}