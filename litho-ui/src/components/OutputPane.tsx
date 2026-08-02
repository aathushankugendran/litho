import { useEffect, useState } from "react";
import type { StreamEvent } from "../types";

interface Props {
  events: StreamEvent[];
  isRunning: boolean;
  label?: string;
  accent?: string;
}

// Plain-language notes on each stat, surfaced on hover so the numbers mean
// something to someone who hasn't worked with transformer internals before.
const STAT_NOTES: Record<string, string> = {
  tokens: "words or word-pieces generated so far",
  "tok/s": "generation speed — tokens produced per second",
  "kv-cache": "memory holding previously computed key/value vectors for reuse",
};

export function OutputPane({ events, isRunning, label = "output", accent = "var(--amber)" }: Props) {
  const text = events.map((e) => e.piece).join("");
  const last = events[events.length - 1];
  const tokensPerSec =
    last && last.elapsed_ms > 0 ? (last.step / (last.elapsed_ms / 1000)).toFixed(2) : "—";

  // A live-ticking clock independent of token arrivals, so the UI visibly
  // moves even during a long gap between tokens (naive mode's first step can
  // take several seconds) instead of appearing frozen.
  const [liveMs, setLiveMs] = useState(0);
  useEffect(() => {
    if (!isRunning) {
      setLiveMs(0);
      return;
    }
    const start = Date.now();
    const interval = setInterval(() => setLiveMs(Date.now() - start), 100);
    return () => clearInterval(interval);
  }, [isRunning]);

  const displayMs = last?.elapsed_ms ?? liveMs;

  return (
    <div className="flex flex-col gap-4 rounded border border-[var(--border)] bg-[var(--panel)] p-5">
      <div>
        <div className="mono mb-1 flex items-center gap-2 text-xs text-[var(--text-dim)]">
          <span style={{ color: accent }}>{label}</span>
          {isRunning && (
            <span className="flex items-center gap-1.5" style={{ color: accent }}>
              <span
                className="h-1.5 w-1.5 animate-pulse rounded-full"
                style={{ background: accent }}
              />
              generating… {(displayMs / 1000).toFixed(1)}s
            </span>
          )}
        </div>
        <div className="min-h-[4rem] whitespace-pre-wrap text-sm leading-relaxed">
          {text || (
            <span className="text-[var(--text-dim)]">
              {isRunning ? "waiting on the first token…" : "Enter a prompt and press generate."}
            </span>
          )}
          {isRunning && (
            <span className="mono ml-0.5 animate-pulse" style={{ color: accent }}>
              ▌
            </span>
          )}
        </div>
      </div>

      <div className="mono grid grid-cols-3 gap-3 border-t border-[var(--border)] pt-3 text-xs">
        <Stat label="tokens" value={last?.step ?? 0} accent={accent} />
        <Stat label="tok/s" value={tokensPerSec} accent={accent} />
        <Stat label="kv-cache" value={`${last?.cache_kb ?? 0} KB`} accent={accent} />
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string | number; accent: string }) {
  return (
    <div className="rounded bg-[var(--bg)] p-2" title={STAT_NOTES[label]}>
      <div className="text-[var(--text-dim)]">{label}</div>
      <div className="text-base" style={{ color: accent }}>
        {value}
      </div>
    </div>
  );
}