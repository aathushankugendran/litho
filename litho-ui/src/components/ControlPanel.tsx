import { useState } from "react";

type RunMode = "cached" | "naive" | "race";

interface Props {
  onRun: (
    mode: RunMode,
    prompt: string,
    maxTokens: number,
    sampled: boolean,
    temperature: number,
    topK: number,
    topP: number
  ) => void;
  isRunning: boolean;
}

// Each mode gets a plain-language explanation, shown when selected, so the
// terms "naive" and "cached" mean something to a first-time viewer before
// they press anything.
const MODE_NOTES: Record<RunMode, string> = {
  cached:
    "Reuses the key/value vectors it already computed for earlier tokens, so each new token only requires computing one token's worth of work.",
  naive:
    "Recomputes every earlier token's key/value vectors from scratch on every single step — correct, but does the same work over and over.",
  race:
    "Runs both approaches on the same prompt at once, so you can watch the difference in speed as it happens.",
};

export function ControlPanel({ onRun, isRunning }: Props) {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [mode, setMode] = useState<RunMode>("cached");
  const [sampled, setSampled] = useState(false);
  const [maxTokens, setMaxTokens] = useState(20);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);

  const modes: { key: RunMode; label: string; accent: string }[] = [
    { key: "cached", label: "cached", accent: "var(--amber)" },
    { key: "naive", label: "naive", accent: "var(--cyan)" },
    { key: "race", label: "race", accent: "var(--text)" },
  ];

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
        <p className="mt-1 text-xs text-[var(--text-dim)]">
          Short factual prompts work best — this is a 1.1B-parameter model, small by modern
          standards.
        </p>
      </div>

      <div>
        <label className="mono mb-1 block text-xs text-[var(--text-dim)]">
          how it handles previous tokens
        </label>
        <div className="grid grid-cols-3 gap-2">
          {modes.map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              className="mono rounded border px-2 py-1.5 text-xs transition-colors"
              style={{
                borderColor: mode === m.key ? m.accent : "var(--border)",
                color: mode === m.key ? m.accent : "var(--text-dim)",
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs leading-relaxed text-[var(--text-dim)]">{MODE_NOTES[mode]}</p>
      </div>

      {mode !== "race" && (
        <div>
          <label className="mono flex items-center gap-2 text-xs text-[var(--text-dim)]">
            <input type="checkbox" checked={sampled} onChange={(e) => setSampled(e.target.checked)} />
            add randomness to word choice
          </label>
          <p className="mt-1 text-xs leading-relaxed text-[var(--text-dim)]">
            {sampled
              ? "Picks from several likely next words instead of always the single most likely one — more varied, less prone to repeating itself."
              : "Currently always picks the single most likely next word. Deterministic, but can get stuck repeating phrases."}
          </p>
        </div>
      )}

      {mode !== "race" && sampled && (
        <div className="flex flex-col gap-3 border-l border-[var(--border)] pl-3">
          <Slider
            label="temperature"
            note="higher = more adventurous word choices"
            value={temperature}
            min={0.1}
            max={2}
            step={0.1}
            onChange={setTemperature}
          />
          <Slider
            label="top_k"
            note="only consider this many top candidates"
            value={topK}
            min={1}
            max={100}
            step={1}
            onChange={setTopK}
          />
          <Slider
            label="top_p"
            note="keep adding candidates until they cover this much probability"
            value={topP}
            min={0.1}
            max={1}
            step={0.05}
            onChange={setTopP}
          />
        </div>
      )}

      <div>
        <label className="mono mb-1 block text-xs text-[var(--text-dim)]">
          words to generate: {maxTokens}
        </label>
        <input
          type="range"
          min={5}
          max={60}
          value={maxTokens}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
          className="w-full accent-[var(--amber)]"
        />
      </div>

      <button
        disabled={isRunning}
        onClick={() => onRun(mode, prompt, maxTokens, sampled, temperature, topK, topP)}
        className="mono rounded px-3 py-2 text-xs font-semibold text-black transition-opacity disabled:opacity-40"
        style={{ background: mode === "naive" ? "var(--cyan)" : "var(--amber)" }}
      >
        {mode === "race" ? "run both, side by side" : "generate"}
      </button>

      <a
        href="https://github.com/aathushankugendran/litho"
        target="_blank"
        rel="noreferrer"
        className="mono border-t border-[var(--border)] pt-4 text-xs text-[var(--text-dim)] underline decoration-dotted underline-offset-4 hover:text-[var(--text)]"
      >
        source code and how it was verified →
      </a>
    </div>
  );
}

function Slider({
  label,
  note,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  note: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="mono mb-0.5 block text-xs text-[var(--text-dim)]">
        {label}: {value}
      </label>
      <p className="mb-1 text-xs text-[var(--text-dim)] opacity-70">{note}</p>
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