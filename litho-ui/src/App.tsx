import { useState } from "react";
import { ModelInfoBar } from "./components/ModelInfoBar";
import { ControlPanel } from "./components/ControlPanel";
import { OutputPane } from "./components/OutputPane";
import { TraceChart } from "./components/TraceChart";
import { RaceSummary } from "./components/RaceSummary";
import { useGeneration } from "./hooks/useGeneration";
import { useModelInfo } from "./hooks/useModelInfo";
import type { StreamEvent } from "./types";

type RunMode = "cached" | "naive" | "race";

export default function App() {
  const info = useModelInfo();
  const { run } = useGeneration();

  const [singleEvents, setSingleEvents] = useState<StreamEvent[]>([]);
  const [isRunningSingle, setIsRunningSingle] = useState(false);
  const [singleAccent, setSingleAccent] = useState("var(--amber)");

  const [naiveEvents, setNaiveEvents] = useState<StreamEvent[]>([]);
  const [cachedEvents, setCachedEvents] = useState<StreamEvent[]>([]);
  const [isRacing, setIsRacing] = useState(false);

  const isRunning = isRunningSingle || isRacing;

  const handleRun = async (
    mode: RunMode,
    prompt: string,
    maxTokens: number,
    sampled: boolean,
    temperature: number,
    topK: number,
    topP: number
  ) => {
    setSingleEvents([]);
    setNaiveEvents([]);
    setCachedEvents([]);

    if (mode === "race") {
      setIsRacing(true);
      try {
        await Promise.all([
          run(prompt, "cached", maxTokens, null, (e) => setCachedEvents((prev) => [...prev, e])),
          run(prompt, "naive", maxTokens, null, (e) => setNaiveEvents((prev) => [...prev, e])),
        ]);
      } finally {
        setIsRacing(false);
      }
      return;
    }

    setSingleAccent(mode === "naive" ? "var(--cyan)" : "var(--amber)");
    setIsRunningSingle(true);
    const sampling = sampled ? { temperature, top_k: topK, top_p: topP } : null;
    try {
      await run(prompt, mode, maxTokens, sampling, (e) => setSingleEvents((prev) => [...prev, e]));
    } finally {
      setIsRunningSingle(false);
    }
  };

  const isRaceView = isRacing || naiveEvents.length > 0 || cachedEvents.length > 0;

  return (
    <div className="min-h-screen">
      <ModelInfoBar info={info} />
      <main className="mx-auto flex max-w-6xl flex-col gap-4 p-4 sm:gap-6 sm:p-6">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-[320px_1fr]">
          <ControlPanel onRun={handleRun} isRunning={isRunning} />

          <div className="flex flex-col gap-6">
            {isRaceView ? (
              <>
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <OutputPane events={cachedEvents} isRunning={isRacing} label="cached" accent="var(--amber)" />
                  <OutputPane events={naiveEvents} isRunning={isRacing} label="naive" accent="var(--cyan)" />
                </div>
                <RaceSummary cachedEvents={cachedEvents} naiveEvents={naiveEvents} />
              </>
            ) : (
              <OutputPane events={singleEvents} isRunning={isRunningSingle} accent={singleAccent} />
            )}
            <TraceChart cachedEvents={cachedEvents} naiveEvents={naiveEvents} />
          </div>
        </div>
      </main>
    </div>
  );
}