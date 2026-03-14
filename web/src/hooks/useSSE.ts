import { useEffect, useRef, useState, useCallback } from "react";
import type { SSEEvent } from "../api/types";

export function useSSE(url: string = "/api/dashboard/events") {
  const [data, setData] = useState<SSEEvent | null>(null);
  const [connected, setConnected] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const [history, setHistory] = useState<number[]>([]);

  const connect = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
    }

    const es = new EventSource(url);
    sourceRef.current = es;

    es.onopen = () => setConnected(true);

    es.onmessage = (event) => {
      try {
        const parsed: SSEEvent = JSON.parse(event.data);
        setData(parsed);

        if (parsed.diagnostics?.average_tokens_per_second !== undefined) {
          setHistory((prev) => {
            const next = [...prev, parsed.diagnostics!.average_tokens_per_second];
            return next.length > 60 ? next.slice(-60) : next;
          });
        }
      } catch {
        // ignore
      }
    };

    es.onerror = () => {
      setConnected(false);
      es.close();
      // Reconnect after 3s
      setTimeout(connect, 3000);
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      sourceRef.current?.close();
    };
  }, [connect]);

  return { data, connected, tokHistory: history };
}
