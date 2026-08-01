import { useState } from "react";
import { ModelInfoBar } from "./components/ModelInfoBar";
import { ControlPanel } from "./components/ControlPanel";
import { OutputPane } from "./components/OutputPane";
import { TraceChart } from "./components/TraceChart";
import { useGeneration } from "./hooks/useGeneration";
import { useModelInfo } from "./hooks/useModelInfo";
import type { Mode, StreamEvent } from "./types";

export default function App() {
  const info = useModelInfo();
  const { events, isRunning, run } = useGeneration();
  const [naiveEvents, setNaiveEvents] = useState<StreamEvent[]>([]);
  const [cachedEvents, setCachedEvents] = useState<StreamEvent[]>([]);

  const handleRun = (
    prompt: string,
    mode: Mode,
    maxTokens: number,
    sampled: boolean,
    temperature: number,
    topK: number,
    topP: number
  ) => {
    setNaiveEvents([]);
    setCachedEvents([]);
    const sampling = sampled ? { temperature, top_k: topK, top_p: topP } : null;
    run(prompt, mode, maxTokens, sampling, (e) => {
      if (mode === "cached") setCachedEvents((prev) => [...prev, e]);
      else setNaiveEvents((prev) => [...prev, e]);
    });
  };

  const handleRace = async (prompt: string, maxTokens: number) => {
    setNaiveEvents([]);
    setCachedEvents([]);
    await Promise.all([
      run(prompt, "cached", maxTokens, null, (e) => setCachedEvents((prev) => [...prev, e])),
      run(prompt, "naive", maxTokens, null, (e) => setNaiveEvents((prev) => [...prev, e])),
    ]);
  };

  const activeEvents = cachedEvents.length > 0 ? cachedEvents : naiveEvents;

  return (
    <div className="min-h-screen">
      <ModelInfoBar info={info} />
      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 p-6 md:grid-cols-[320px_1fr]">
        <ControlPanel onRun={handleRun} onRace={handleRace} isRunning={isRunning} />
        <div className="flex flex-col gap-6">
          <OutputPane events={events.length > 0 ? events : activeEvents} />
          <TraceChart cachedEvents={cachedEvents} naiveEvents={naiveEvents} />
        </div>
      </main>
    </div>
  );
}