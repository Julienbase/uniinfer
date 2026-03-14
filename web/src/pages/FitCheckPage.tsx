import { useState } from "react";
import { runFitCheck } from "../api/client";
import { useAliases, useDevices } from "../api/hooks";
import type { DashboardFitCheckResponse } from "../api/types";

export function FitCheckPage() {
  const { data: devices = [] } = useDevices();
  const { data: aliases = [] } = useAliases();
  const [modelId, setModelId] = useState("");
  const [quantization, setQuantization] = useState("q4_k_m");
  const [contextLength, setContextLength] = useState(4096);
  const [isChecking, setIsChecking] = useState(false);
  const [result, setResult] = useState<DashboardFitCheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Use best available device for display (active one if exists, otherwise first)
  const bestDevice = devices.find((d) => d.is_active) || devices[0];
  const deviceName = bestDevice?.name || "No device detected";
  const freeMemory = bestDevice?.free_memory_gb ?? 0;

  const quants = ["f16", "q8_0", "q6_k", "q5_k_m", "q5_0", "q4_k_m", "q4_0", "q3_k_m", "q2_k"];

  const handleCheck = async () => {
    if (!modelId.trim() || isChecking) return;
    setIsChecking(true);
    setError(null);
    setResult(null);

    try {
      const res = await runFitCheck({
        model_id: modelId.trim(),
        quantization,
        context_length: contextLength,
      });
      if (res.error) {
        setError(res.error);
      } else {
        setResult(res);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Fit check failed");
    } finally {
      setIsChecking(false);
    }
  };

  const selectAlias = (alias: string) => {
    const info = aliases.find((a) => a.alias === alias);
    if (info) {
      setModelId(info.repo_id);
      setQuantization(info.default_quant);
      setContextLength(info.default_context_length);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-semibold text-text-primary tracking-tight">Fit Check</h2>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-card border border-border text-text-muted font-[JetBrains_Mono]">
          dry run
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-4">
        {/* Input panel */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border bg-bg-secondary/80">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${bestDevice ? "bg-success" : "bg-warning"}`} />
              <span className="text-sm text-text-secondary">{deviceName}</span>
              {bestDevice && (
                <>
                  <span className="text-text-muted text-[10px]">·</span>
                  <span className="text-[10px] text-text-muted font-[JetBrains_Mono]">
                    {freeMemory.toFixed(1)} GB free
                  </span>
                </>
              )}
            </div>
          </div>

          <div className="p-4 space-y-4">
            {/* Quick select from aliases */}
            {aliases.length > 0 && (
              <div>
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1.5">
                  Quick Select
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {aliases.slice(0, 8).map((a) => (
                    <button
                      key={a.alias}
                      onClick={() => selectAlias(a.alias)}
                      className="px-2.5 py-1 text-[11px] rounded-md bg-bg-card border border-border text-text-secondary hover:text-text-primary hover:border-border-bright transition-all cursor-pointer"
                    >
                      {a.display_name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div>
              <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1.5">
                Model ID (HuggingFace repo or alias)
              </label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="e.g., TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
                className="w-full bg-bg-input border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted/50 font-[JetBrains_Mono] focus:outline-none focus:border-accent/50"
              />
            </div>

            <div className="flex gap-4">
              <div className="flex-1">
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                  Quantization
                </label>
                <select
                  value={quantization}
                  onChange={(e) => setQuantization(e.target.value)}
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-[JetBrains_Mono] focus:outline-none focus:border-accent/50"
                >
                  {quants.map((q) => (
                    <option key={q} value={q}>{q}</option>
                  ))}
                </select>
              </div>
              <div className="flex-1">
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                  Context Length
                </label>
                <input
                  type="number"
                  min={512}
                  max={131072}
                  step={512}
                  value={contextLength}
                  onChange={(e) => setContextLength(parseInt(e.target.value) || 4096)}
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-[JetBrains_Mono] focus:outline-none focus:border-accent/50"
                />
              </div>
            </div>

            <button
              onClick={handleCheck}
              disabled={!modelId.trim() || isChecking}
              className="w-full py-2.5 rounded-xl bg-accent/15 border border-accent/30 text-accent text-sm font-medium hover:bg-accent/25 transition-all cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {isChecking ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  Checking...
                </span>
              ) : (
                "Check Fit"
              )}
            </button>
          </div>
        </div>

        {/* Result panel */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border">
            <span className="text-xs text-text-secondary font-medium">Fit Report</span>
          </div>

          <div className="p-4">
            {error && (
              <div className="bg-danger-dim border border-danger/30 rounded-lg px-4 py-3 mb-3">
                <p className="text-sm text-danger">{error}</p>
              </div>
            )}

            {!result && !error && !isChecking && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="w-10 h-10 rounded-xl bg-bg-card border border-border flex items-center justify-center mb-3">
                  <svg className="w-5 h-5 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-text-muted text-xs">Select a model to check if it fits</p>
              </div>
            )}

            {isChecking && (
              <div className="flex flex-col items-center justify-center py-16">
                <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mb-3" />
                <p className="text-text-muted text-xs">Querying model size...</p>
              </div>
            )}

            {result && (
              <div className="space-y-4 animate-fade-in">
                {/* Fit status */}
                <div
                  className={`rounded-xl px-5 py-4 text-center border ${
                    result.fits
                      ? "bg-success-dim border-success/20"
                      : "bg-danger-dim border-danger/20"
                  }`}
                >
                  <p className={`text-lg font-bold ${result.fits ? "text-success" : "text-danger"}`}>
                    {result.fits ? "FITS" : "DOES NOT FIT"}
                  </p>
                  <p className="text-xs text-text-muted mt-1">{result.device_name}</p>
                </div>

                {/* Memory breakdown */}
                <div className="space-y-1.5">
                  <p className="text-[10px] uppercase tracking-wider text-text-muted font-semibold">
                    Memory Breakdown
                  </p>
                  {[
                    { label: "Model Size", value: result.model_size_gb, unit: "GB" },
                    { label: "Overhead (KV cache + runtime)", value: result.overhead_gb, unit: "GB" },
                    { label: "Available Memory", value: result.available_memory_gb, unit: "GB" },
                    {
                      label: "Headroom",
                      value: result.headroom_gb,
                      unit: "GB",
                      color: (result.headroom_gb ?? 0) >= 0 ? "text-success" : "text-danger",
                    },
                  ].map((row) => (
                    <div key={row.label} className="flex items-center justify-between bg-bg-card rounded-lg px-3 py-2">
                      <span className="text-xs text-text-secondary">{row.label}</span>
                      <span className={`text-sm font-[JetBrains_Mono] ${row.color || "text-text-primary"}`}>
                        {row.value != null ? `${row.value} ${row.unit}` : "—"}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Warnings */}
                {result.warnings.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-[10px] uppercase tracking-wider text-warning font-semibold">
                      Warnings
                    </p>
                    {result.warnings.map((w, i) => (
                      <div key={i} className="bg-warning-dim border border-warning/20 rounded-lg px-3 py-2">
                        <p className="text-xs text-text-secondary">{w}</p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Alternatives */}
                {result.alternatives.length > 0 && (
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-semibold mb-1.5">
                      Quantization Alternatives
                    </p>
                    <div className="border border-border rounded-lg overflow-hidden">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-bg-card/50">
                            <th className="text-left px-3 py-1.5 text-text-muted font-medium">Quant</th>
                            <th className="text-right px-3 py-1.5 text-text-muted font-medium">Est. Size</th>
                            <th className="text-center px-3 py-1.5 text-text-muted font-medium">Fits?</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.alternatives.map((alt) => (
                            <tr
                              key={alt.quantization}
                              className={`border-t border-border ${
                                alt.quantization === result.recommended_quantization
                                  ? "bg-accent/5"
                                  : ""
                              }`}
                            >
                              <td className="px-3 py-1.5 font-[JetBrains_Mono] text-text-primary">
                                {alt.quantization}
                                {alt.quantization === result.recommended_quantization && (
                                  <span className="ml-1.5 text-[9px] text-accent uppercase">rec</span>
                                )}
                              </td>
                              <td className="text-right px-3 py-1.5 font-[JetBrains_Mono] text-text-secondary">
                                {alt.estimated_size_gb} GB
                              </td>
                              <td className="text-center px-3 py-1.5">
                                <span className={alt.fits ? "text-success" : "text-danger"}>
                                  {alt.fits ? "Yes" : "No"}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
