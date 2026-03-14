import { useState } from "react";
import { dashboardGenerate } from "../api/client";
import { useStatus } from "../api/hooks";
import type { DashboardGenerateResponse } from "../api/types";

export function GeneratePage() {
  const { data: status } = useStatus();
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(512);
  const [temperature, setTemperature] = useState(0.7);
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<DashboardGenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const modelLoaded = status?.loaded ?? false;
  const modelName = status?.model || "—";

  const handleGenerate = async () => {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setError(null);
    setResult(null);

    try {
      const res = await dashboardGenerate({
        prompt: prompt.trim(),
        max_tokens: maxTokens,
        temperature,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-semibold text-text-primary tracking-tight">Generate</h2>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-card border border-border text-text-muted font-[JetBrains_Mono]">
          one-shot
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
        {/* Input panel */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border bg-bg-secondary/80">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${modelLoaded ? "bg-success" : "bg-danger"}`} />
              <span className="text-sm text-text-secondary font-[JetBrains_Mono]">{modelName}</span>
            </div>
          </div>

          <div className="p-4 space-y-4">
            <div>
              <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1.5">
                Prompt
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt here..."
                rows={6}
                className="w-full bg-bg-input border border-border rounded-xl px-4 py-3 text-sm text-text-primary placeholder-text-muted/50 resize-none focus:outline-none focus:border-accent/50"
              />
            </div>

            <div className="flex gap-4">
              <div className="flex-1">
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                  Temperature: {temperature.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-accent h-1"
                />
              </div>
              <div className="w-28">
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                  Max Tokens
                </label>
                <input
                  type="number"
                  min={1}
                  max={4096}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 512)}
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-[JetBrains_Mono] focus:outline-none focus:border-accent/50"
                />
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={!prompt.trim() || !modelLoaded || isGenerating}
              className="w-full py-2.5 rounded-xl bg-accent/15 border border-accent/30 text-accent text-sm font-medium hover:bg-accent/25 transition-all cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {isGenerating ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  Generating...
                </span>
              ) : (
                "Generate"
              )}
            </button>
          </div>
        </div>

        {/* Result panel */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border">
            <span className="text-xs text-text-secondary font-medium">Output</span>
          </div>

          <div className="p-4">
            {error && (
              <div className="bg-danger-dim border border-danger/30 rounded-lg px-4 py-3 mb-3">
                <p className="text-sm text-danger">{error}</p>
              </div>
            )}

            {!result && !error && !isGenerating && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="w-10 h-10 rounded-xl bg-bg-card border border-border flex items-center justify-center mb-3">
                  <svg className="w-5 h-5 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                  </svg>
                </div>
                <p className="text-text-muted text-xs">Enter a prompt and click Generate</p>
              </div>
            )}

            {isGenerating && (
              <div className="flex items-center justify-center py-12">
                <div className="flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            )}

            {result && (
              <div className="space-y-3 animate-fade-in">
                <div className="bg-bg-card border border-border rounded-xl p-4">
                  <p className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed break-words">
                    {result.text}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-bg-card rounded-lg px-3 py-2 text-center">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider">Tokens</p>
                    <p className="text-sm font-[JetBrains_Mono] text-text-primary">
                      {result.prompt_tokens} + {result.completion_tokens}
                    </p>
                  </div>
                  <div className="bg-bg-card rounded-lg px-3 py-2 text-center">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider">Speed</p>
                    <p className="text-sm font-[JetBrains_Mono] text-accent">
                      {result.tokens_per_second} tok/s
                    </p>
                  </div>
                  <div className="bg-bg-card rounded-lg px-3 py-2 text-center">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider">Time</p>
                    <p className="text-sm font-[JetBrains_Mono] text-text-primary">
                      {result.elapsed_seconds}s
                    </p>
                  </div>
                  <div className="bg-bg-card rounded-lg px-3 py-2 text-center">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider">Total</p>
                    <p className="text-sm font-[JetBrains_Mono] text-text-primary">
                      {result.total_tokens} tok
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
