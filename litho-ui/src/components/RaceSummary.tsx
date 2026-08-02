import type { StreamEvent } from "../types";

interface Props {
  cachedEvents: StreamEvent[];
  naiveEvents: StreamEvent[];
}

export function RaceSummary({ cachedEvents, naiveEvents }: Props) {
  if (cachedEvents.length === 0 && naiveEvents.length === 0) return null;

  const cachedLast = cachedEvents[cachedEvents.length - 1];
  const naiveLast = naiveEvents[naiveEvents.length - 1];

  const cachedMs = cachedLast?.elapsed_ms ?? 0;
  const naiveMs = naiveLast?.elapsed_ms ?? 0;
  const speedup = cachedMs > 0 ? naiveMs / cachedMs : 0;

  const cacheKb = cachedLast?.cache_kb ?? 0;
  const step = Math.max(cachedLast?.step ?? 0, naiveLast?.step ?? 0);

  return (
    <div className="rounded border border-[var(--border)] bg-[var(--panel)] p-5">
      <div className="mono mb-1 text-xs text-[var(--text-dim)]">what the cache bought</div>
      <p className="mb-4 text-xs leading-relaxed text-[var(--text-dim)]">
        Both runs produce the same text — the cache changes only how much work gets repeated, never
        the result.
      </p>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded bg-[var(--bg)] p-3">
          <div className="mono text-xs text-[var(--text-dim)]">cached is faster by</div>
          <div className="mono text-2xl text-[var(--amber)]">
            {speedup > 0 ? `${speedup.toFixed(2)}x` : "—"}
          </div>
        </div>
        <div className="rounded bg-[var(--bg)] p-3">
          <div className="mono text-xs text-[var(--text-dim)]">time taken</div>
          <div className="mono text-lg">
            <span className="text-[var(--cyan)]">{(naiveMs / 1000).toFixed(1)}s</span>
            <span className="text-[var(--text-dim)]"> vs </span>
            <span className="text-[var(--amber)]">{(cachedMs / 1000).toFixed(1)}s</span>
          </div>
          <div className="mono mt-0.5 text-xs text-[var(--text-dim)]">naive vs cached</div>
        </div>
        <div className="rounded bg-[var(--bg)] p-3">
          <div className="mono text-xs text-[var(--text-dim)]">memory the cache used</div>
          <div className="mono text-lg text-[var(--amber)]">{cacheKb} KB</div>
          <div className="mono mt-0.5 text-xs text-[var(--text-dim)]">naive used none</div>
        </div>
      </div>

      {step > 0 && (
        <p className="mt-4 text-xs leading-relaxed text-[var(--text-dim)]">
          That speed difference is the whole tradeoff: by spending {cacheKb}KB of memory storing what
          it already worked out, the cached run avoids redoing {step - 1} tokens' worth of attention
          math on every single step. The naive run keeps no memory at all, and pays for it in time —
          a gap that widens the longer generation runs.
        </p>
      )}
    </div>
  );
}