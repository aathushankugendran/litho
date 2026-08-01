import type { StreamEvent } from "../types";

export function OutputPane({ events }: { events: StreamEvent[] }) {
  const text = events.map((e) => e.piece).join("");
  const last = events[events.length - 1];
  const tokensPerSec = last && last.elapsed_ms > 0 ? (last.step / (last.elapsed_ms / 1000)).toFixed(2) : "—";

  return (
    <div className="flex flex-col gap-4 rounded border border-[var(--border)] bg-[var(--panel)] p-5">
      <div>
        <div className="mono mb-1 text-xs text-[var(--text-dim)]">output</div>
        <div className="min-h-[4rem] whitespace-pre-wrap text-sm leading-relaxed">
          {text || <span className="text-[var(--text-dim)]">— waiting —</span>}
        </div>
      </div>

      <div className="mono grid grid-cols-3 gap-3 border-t border-[var(--border)] pt-3 text-xs">
        <Stat label="tokens" value={last?.step ?? 0} />
        <Stat label="tok/s" value={tokensPerSec} />
        <Stat label="kv-cache" value={`${last?.cache_kb ?? 0} KB`} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded bg-[var(--bg)] p-2">
      <div className="text-[var(--text-dim)]">{label}</div>
      <div className="text-base text-[var(--amber)]">{value}</div>
    </div>
  );
}