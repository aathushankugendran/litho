import { useEffect, useState } from "react";
import type { ModelInfo } from "../types";

const API_BASE = "http://localhost:3000";

export function useModelInfo() {
  const [info, setInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/model-info`)
      .then((r) => r.json())
      .then(setInfo)
      .catch((err) => console.error("failed to load model info", err));
  }, []);

  return info;
}