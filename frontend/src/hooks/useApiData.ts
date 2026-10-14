import { useEffect, useState } from "react";
import { api, type Health, type ModelInfo } from "../lib/api";

/** Health and model info, fetched once for the whole page. */
export function useApiData() {
  const [health, setHealth] = useState<Health | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.health(), api.modelInfo()])
      .then(([h, i]) => {
        if (cancelled) return;
        setHealth(h);
        setInfo(i);
      })
      .catch(() => !cancelled && setOffline(true));
    return () => {
      cancelled = true;
    };
  }, []);

  return { health, info, offline };
}
