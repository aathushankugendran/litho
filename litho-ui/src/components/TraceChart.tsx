import type { StreamEvent } from "../types";

interface Props {
  cachedEvents: StreamEvent[];
  naiveEvents: StreamEvent[];
}

const WIDTH = 640;
const HEIGHT = 200;
const PAD_LEFT = 56;
const PAD_RIGHT = 16;
const PAD_TOP = 16;
const PAD_BOTTOM = 28;

const PLOT_W = WIDTH - PAD_LEFT - PAD_RIGHT;
const PLOT_H = HEIGHT - PAD_TOP - PAD_BOTTOM;

function pointFor(e: StreamEvent, maxStep: number, maxMs: number) {
  const x = PAD_LEFT + (e.step / Math.max(maxStep, 1)) * PLOT_W;
  const y = PAD_TOP + PLOT_H - (e.elapsed_ms / Math.max(maxMs, 1)) * PLOT_H;
  return { x, y };
}

function pathFor(events: StreamEvent[], maxStep: number, maxMs: number) {
  if (events.length === 0) return "";
  return events
    .map((e, i) => {
      const { x, y } = pointFor(e, maxStep, maxMs);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function formatMs(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export function TraceChart({ cachedEvents, naiveEvents }: Props) {
  const allEvents = [...cachedEvents, ...naiveEvents];
  const maxStep = Math.max(...allEvents.map((e) => e.step), 1);
  const maxMs = Math.max(...allEvents.map((e) => e.elapsed_ms), 1000);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * maxMs);
  const xTickCount = Math.min(maxStep, 6);
  const xTicks = Array.from({ length: xTickCount + 1 }, (_, i) =>
    Math.round((i / xTickCount) * maxStep)
  );

  const cachedLast = cachedEvents[cachedEvents.length - 1];
  const naiveLast = naiveEvents[naiveEvents.length - 1];

  return (
    <div className="rounded border border-[var(--border)] bg-[var(--panel)] p-4">
      <div className="mono mb-2 flex items-center justify-between text-xs text-[var(--text-dim)]">
        <span>elapsed time per token</span>
        <span className="flex gap-4">
          <span className="text-[var(--cyan)]">■ naive</span>
          <span className="text-[var(--amber)]">■ cached</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full">
        {yTicks.map((tick) => {
          const y = PAD_TOP + PLOT_H - (tick / maxMs) * PLOT_H;
          return (
            <g key={tick}>
              <line x1={PAD_LEFT} x2={WIDTH - PAD_RIGHT} y1={y} y2={y} stroke="var(--border)" strokeWidth={1} />
              <text x={PAD_LEFT - 8} y={y + 3} textAnchor="end" fontSize={10} fill="var(--text-dim)" className="mono">
                {formatMs(tick)}
              </text>
            </g>
          );
        })}

        {xTicks.map((tick) => {
          const x = PAD_LEFT + (tick / Math.max(maxStep, 1)) * PLOT_W;
          return (
            <g key={tick}>
              <line
                x1={x}
                x2={x}
                y1={PAD_TOP + PLOT_H}
                y2={PAD_TOP + PLOT_H + 4}
                stroke="var(--text-dim)"
                strokeWidth={1}
              />
              <text
                x={x}
                y={PAD_TOP + PLOT_H + 16}
                textAnchor="middle"
                fontSize={10}
                fill="var(--text-dim)"
                className="mono"
              >
                {tick}
              </text>
            </g>
          );
        })}

        <line x1={PAD_LEFT} x2={PAD_LEFT} y1={PAD_TOP} y2={PAD_TOP + PLOT_H} stroke="var(--text-dim)" strokeWidth={1} />
        <line
          x1={PAD_LEFT}
          x2={WIDTH - PAD_RIGHT}
          y1={PAD_TOP + PLOT_H}
          y2={PAD_TOP + PLOT_H}
          stroke="var(--text-dim)"
          strokeWidth={1}
        />

        <text x={PAD_LEFT + PLOT_W / 2} y={HEIGHT - 2} textAnchor="middle" fontSize={10} fill="var(--text-dim)" className="mono">
          token step
        </text>

        <path d={pathFor(naiveEvents, maxStep, maxMs)} fill="none" stroke="var(--cyan)" strokeWidth={2} />
        <path d={pathFor(cachedEvents, maxStep, maxMs)} fill="none" stroke="var(--amber)" strokeWidth={2} />

        {/* finish markers -- a filled dot plus elapsed time at the last point of each trace */}
        {naiveLast && (
          <g>
            <circle cx={pointFor(naiveLast, maxStep, maxMs).x} cy={pointFor(naiveLast, maxStep, maxMs).y} r={3.5} fill="var(--cyan)" />
            <text
              x={pointFor(naiveLast, maxStep, maxMs).x}
              y={pointFor(naiveLast, maxStep, maxMs).y - 8}
              textAnchor="end"
              fontSize={10}
              fill="var(--cyan)"
              className="mono"
            >
              {formatMs(naiveLast.elapsed_ms)}
            </text>
          </g>
        )}
        {cachedLast && (
          <g>
            <circle cx={pointFor(cachedLast, maxStep, maxMs).x} cy={pointFor(cachedLast, maxStep, maxMs).y} r={3.5} fill="var(--amber)" />
            <text
              x={pointFor(cachedLast, maxStep, maxMs).x}
              y={pointFor(cachedLast, maxStep, maxMs).y + 14}
              textAnchor="end"
              fontSize={10}
              fill="var(--amber)"
              className="mono"
            >
              {formatMs(cachedLast.elapsed_ms)}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}