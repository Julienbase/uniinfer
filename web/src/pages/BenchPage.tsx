import { useState } from "react";
import { runBenchmark } from "../api/client";
import { useStatus } from "../api/hooks";
import type { DashboardBenchResponse } from "../api/types";

export function BenchPage() {
  const { data: status } = useStatus();
  const [prompt, setPrompt] = useState("Explain the theory of general relativity in simple terms.");
  const [maxTokens, setMaxTokens] = useState(128);
  const [runs, setRuns] = useState(3);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<DashboardBenchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const modelLoaded = status?.loaded ?? false;
  const modelName = status?.model || "—";

  const handleBench = async () => {
    if (isRunning) return;
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const res = await runBenchmark({ prompt, max_tokens: maxTokens, runs });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-semibold text-text-primary tracking-tight">Benchmark</h2>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-card border border-border text-text-muted font-[JetBrains_Mono]">
          tok/s
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-4">
        {/* Config panel */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border bg-bg-secondary/80">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${modelLoaded ? "bg-success" : "bg-danger"}`} />
              <span className="text-sm text-text-secondary font-[JetBrains_Mono]">{modelName}</span>
              {status?.quantization && (
                <>
                  <span className="text-text-muted text-[10px]">·</span>
                  <span className="text-[10px] text-text-muted font-[JetBrains_Mono]">{status.quantization}</span>
                </>
              )}
              {status?.device_name && (
                <>
                  <span className="text-text-muted text-[10px]">·</span>
                  <span className="text-[10px] text-text-muted">{status.device_name}</span>
                </>
              )}
            </div>
          </div>

          <div className="p-4 space-y-4">
            <div>
              <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1.5">
                Benchmark Prompt
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                className="w-full bg-bg-input border border-border rounded-xl px-4 py-3 text-sm text-text-primary placeholder-text-muted/50 resize-none focus:outline-none focus:border-accent/50"
              />
            </div>

            <div className="flex gap-4">
              <div className="flex-1">
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                  Max Tokens
                </label>
                <input
                  type="number"
                  min={16}
                  max={2048}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 128)}
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-[JetBrains_Mono] focus:outline-none focus:border-accent/50"
                />
              </div>
              <div className="flex-1">
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                  Runs
                </label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={runs}
                  onChange={(e) => setRuns(parseInt(e.target.value) || 3)}
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-[JetBrains_Mono] focus:outline-none focus:border-accent/50"
                />
              </div>
            </div>

            <p className="text-[10px] text-text-muted leading-relaxed">
              Runs inference {runs} time{runs > 1 ? "s" : ""} with temperature=0 (greedy) to measure consistent throughput.
            </p>

            <button
              onClick={handleBench}
              disabled={!modelLoaded || isRunning}
              className="w-full py-2.5 rounded-xl bg-accent/15 border border-accent/30 text-accent text-sm font-medium hover:bg-accent/25 transition-all cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {isRunning ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  Running benchmark...
                </span>
              ) : (
                "Run Benchmark"
              )}
            </button>
          </div>
        </div>

        {/* Results panel */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border">
            <span className="text-xs text-text-secondary font-medium">Results</span>
          </div>

          <div className="p-4">
            {error && (
              <div className="bg-danger-dim border border-danger/30 rounded-lg px-4 py-3 mb-3">
                <p className="text-sm text-danger">{error}</p>
              </div>
            )}

            {!result && !error && !isRunning && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="w-10 h-10 rounded-xl bg-bg-card border border-border flex items-center justify-center mb-3">
                  <svg className="w-5 h-5 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                  </svg>
                </div>
                <p className="text-text-muted text-xs">Click Run Benchmark to start</p>
              </div>
            )}

            {isRunning && (
              <div className="flex flex-col items-center justify-center py-16">
                <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mb-3" />
                <p className="text-text-muted text-xs">Running {runs} inference passes...</p>
              </div>
            )}

            {result && (
              <div className="space-y-4 animate-fade-in">
                {/* Summary cards */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-accent/8 border border-accent/15 rounded-xl px-4 py-3 text-center">
                    <p className="text-[10px] text-accent/70 uppercase tracking-wider mb-1">Average</p>
                    <p className="text-xl font-bold font-[JetBrains_Mono] text-accent">
                      {result.average_tokens_per_second}
                    </p>
                    <p className="text-[10px] text-accent/60">tok/s</p>
                  </div>
                  <div className="bg-success-dim border border-success/20 rounded-xl px-4 py-3 text-center">
                    <p className="text-[10px] text-success/70 uppercase tracking-wider mb-1">Peak</p>
                    <p className="text-xl font-bold font-[JetBrains_Mono] text-success">
                      {result.peak_tokens_per_second}
                    </p>
                    <p className="text-[10px] text-success/60">tok/s</p>
                  </div>
                </div>

                {/* Per-run results */}
                <div className="space-y-1.5">
                  <p className="text-[10px] uppercase tracking-wider text-text-muted font-semibold">
                    Individual Runs
                  </p>
                  {result.runs.map((run) => {
                    const pct = result.peak_tokens_per_second > 0
                      ? (run.tokens_per_second / result.peak_tokens_per_second) * 100
                      : 0;
                    return (
                      <div key={run.run_number} className="bg-bg-card rounded-lg px-3 py-2">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-xs text-text-secondary font-[JetBrains_Mono]">
                            Run {run.run_number}
                          </span>
                          <div className="flex items-center gap-3">
                            <span className="text-[10px] text-text-muted font-[JetBrains_Mono]">
                              {run.tokens} tok in {run.elapsed_seconds}s
                            </span>
                            <span className="text-sm font-bold font-[JetBrains_Mono] text-accent">
                              {run.tokens_per_second} tok/s
                            </span>
                          </div>
                        </div>
                        <div className="h-1 bg-bg-primary rounded-full overflow-hidden">
                          <div
                            className="h-full bg-accent/50 rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Meta info */}
                <div className="flex items-center gap-2 text-[10px] text-text-muted font-[JetBrains_Mono]">
                  <span>{result.model}</span>
                  <span>·</span>
                  <span>{result.quantization}</span>
                  <span>·</span>
                  <span>{result.device}</span>
                  <span>·</span>
                  <span>{result.total_tokens} total tokens</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
