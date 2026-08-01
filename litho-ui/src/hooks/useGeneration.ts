import { useCallback, useRef, useState } from "react";
import type { Mode, SamplingConfig, StreamEvent } from "../types";

const API_BASE = "http://localhost:3000";

export function useGeneration() {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (
      prompt: string,
      mode: Mode,
      maxTokens: number,
      sampling: SamplingConfig | null,
      onEvent?: (e: StreamEvent) => void
    ) => {
      setEvents([]);
      setIsRunning(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch(`${API_BASE}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, mode, max_tokens: maxTokens, sampling }),
          signal: controller.signal,
        });

        if (!res.body) throw new Error("no response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            const line = frame.split("\n").find((l) => l.startsWith("data: "));
            if (!line) continue;
            const json = line.slice("data: ".length);
            const parsed: StreamEvent = JSON.parse(json);
            setEvents((prev) => [...prev, parsed]);
            onEvent?.(parsed);
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") console.error(err);
      } finally {
        setIsRunning(false);
      }
    },
    []
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsRunning(false);
  }, []);

  return { events, isRunning, run, stop };
}