import { useCallback } from "react";
import type { Mode, SamplingConfig, StreamEvent } from "../types";

const API_BASE = "http://localhost:3000";

// Stateless by design: each call to run() reads its own response stream and
// reports events only through the callback. Keeping no internal state here
// means two concurrent calls (e.g. a naive vs cached race) never interfere
// with each other -- each caller manages its own events/isRunning state.
export function useGeneration() {
  const run = useCallback(
    async (
      prompt: string,
      mode: Mode,
      maxTokens: number,
      sampling: SamplingConfig | null,
      onEvent: (e: StreamEvent) => void,
      signal?: AbortSignal
    ) => {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, mode, max_tokens: maxTokens, sampling }),
        signal,
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
          onEvent(parsed);
        }
      }
    },
    []
  );

  return { run };
}