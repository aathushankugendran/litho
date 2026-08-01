import type { StreamEvent } from "../types";

interface Props {
  cachedEvents: StreamEvent[];
  naiveEvents: StreamEvent[];
}

const WIDTH = 640;
const HEIGHT = 180;
const PADDING = 24;

function pathFor(events: StreamEvent[], maxStep: number, maxMs: number) {
  if (events.length === 0) return "";
  return events
    .map((e, i) => {
      const x = PADDING + (e.step / Math.max(maxStep, 1)) * (WIDTH - PADDING * 2);
      const y = HEIGHT - PADDING - (e.elapsed_ms / Math.max(maxMs, 1)) * (HEIGHT - PADDING * 2);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

export function TraceChart({ cachedEvents, naiveEvents }: Props) {
  const allEvents = [...cachedEvents, ...naiveEvents];
  const maxStep = Math.max(...allEvents.map((e) => e.step), 1);
  const maxMs = Math.max(...allEvents.map((e) => e.elapsed_ms), 1000);

  return (
    <div className="rounded border border-[var(--border)] bg-[var(--panel)] p-4">
      <div className="mono mb-2 flex items-center justify-between text-xs text-[var(--text-dim)]">
        <span>elapsed time per token (ms)</span>
        <span className="flex gap-4">
          <span className="text-[var(--cyan)]">■ naive</span>
          <span className="text-[var(--amber)]">■ cached</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full">
        {/* grid lines */}
        {[0.25, 0.5, 0.75].map((f) => (
          <line
            key={f}
            x1={PADDING}
            x2={WIDTH - PADDING}
            y1={HEIGHT - PADDING - f * (HEIGHT - PADDING * 2)}
            y2={HEIGHT - PADDING - f * (HEIGHT - PADDING * 2)}
            stroke="var(--border)"
            strokeWidth={1}
          />
        ))}
        <path d={pathFor(naiveEvents, maxStep, maxMs)} fill="none" stroke="var(--cyan)" strokeWidth={2} />
        <path d={pathFor(cachedEvents, maxStep, maxMs)} fill="none" stroke="var(--amber)" strokeWidth={2} />
      </svg>
    </div>
  );
}