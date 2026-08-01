import { useState } from "react";
import type { Mode } from "../types";

interface Props {
  onRun: (prompt: string, mode: Mode, maxTokens: number, sampled: boolean, temp: number, topK: number, topP: number) => void;
  onRace: (prompt: string, maxTokens: number) => void;
  isRunning: boolean;
}

export function ControlPanel({ onRun, onRace, isRunning }: Props) {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [mode, setMode] = useState<Mode>("cached");
  const [sampled, setSampled] = useState(false);
  const [maxTokens, setMaxTokens] = useState(20);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);

  return (
    <div className="flex flex-col gap-5 rounded border border-[var(--border)] bg-[var(--panel)] p-5">
      <div>
        <label className="mono mb-1 block text-xs text-[var(--text-dim)]">prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="mono w-full resize-none rounded border border-[var(--border)] bg-[var(--bg)] p-2 text-sm text-[var(--text)] outline-none focus:border-[var(--amber)]"
        />
      </div>

      <div>
        <label className="mono mb-1 block text-xs text-[var(--text-dim)]">mode</label>
        <div className="flex gap-2">
          {(["cached", "naive"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`mono flex-1 rounded border px-3 py-1.5 text-xs transition-colors ${
                mode === m
                  ? "border-[var(--amber)] text-[var(--amber)]"
                  : "border-[var(--border)] text-[var(--text-dim)] hover:text-[var(--text)]"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <label className="mono flex items-center gap-2 text-xs text-[var(--text-dim)]">
        <input type="checkbox" checked={sampled} onChange={(e) => setSampled(e.target.checked)} />
        sample (temperature / top-k / top-p) instead of greedy
      </label>

      {sampled && (
        <div className="flex flex-col gap-3 border-l border-[var(--border)] pl-3">
          <Slider label="temperature" value={temperature} min={0.1} max={2} step={0.1} onChange={setTemperature} />
          <Slider label="top_k" value={topK} min={1} max={100} step={1} onChange={setTopK} />
          <Slider label="top_p" value={topP} min={0.1} max={1} step={0.05} onChange={setTopP} />
        </div>
      )}

      <div>
        <label className="mono mb-1 block text-xs text-[var(--text-dim)]">max tokens: {maxTokens}</label>
        <input
          type="range"
          min={5}
          max={60}
          value={maxTokens}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
          className="w-full accent-[var(--amber)]"
        />
      </div>

      <div className="flex gap-2">
        <button
          disabled={isRunning}
          onClick={() => onRun(prompt, mode, maxTokens, sampled, temperature, topK, topP)}
          className="mono flex-1 rounded bg-[var(--amber)] px-3 py-2 text-xs font-semibold text-black transition-opacity disabled:opacity-40"
        >
          generate
        </button>
        <button
          disabled={isRunning}
          onClick={() => onRace(prompt, Math.min(maxTokens, 20))}
          className="mono flex-1 rounded border border-[var(--cyan)] px-3 py-2 text-xs font-semibold text-[var(--cyan)] transition-opacity disabled:opacity-40"
        >
          race naive vs cached
        </button>
      </div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="mono mb-1 block text-xs text-[var(--text-dim)]">
        {label}: {value}
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--amber)]"
      />
    </div>
  );
}