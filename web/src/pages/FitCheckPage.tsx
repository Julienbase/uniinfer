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

  const bestDevice = devices.find((d) => d.is_active) || devices[0];
  const deviceName = bestDevice?.name || "No device detected";
  const freeMemory = bestDevice?.free_memory_gb ?? 0;
  const totalMemory = bestDevice?.total_memory_gb ?? 0;
  const usedMemory = totalMemory - freeMemory;
  const usedPct = totalMemory > 0 ? (usedMemory / totalMemory) * 100 : 0;

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
    <div className="space-y-5">
      {/* Page header */}
      <div className="animate-slide-up" style={{ "--delay": "0ms" } as React.CSSProperties}>
        <div className="flex items-center gap-3">
          <svg viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" strokeWidth="1.5" className="w-6 h-6">
            <circle cx="12" cy="12" r="9" />
            <circle cx="12" cy="12" r="3.5" />
            <path d="M12 3v3M12 18v3M3 12h3M18 12h3" strokeLinecap="round" />
          </svg>
          <h2 className="font-display font-bold text-xl text-text-primary tracking-tight">Fit Check</h2>
          <span className="text-[10px] px-2 py-0.5 rounded-md bg-accent-glow text-accent font-mono border border-accent/20 animate-pulse">
            pre-download
          </span>
        </div>
        <p className="text-sm text-text-muted mt-1 ml-9">
          Will this model fit on your hardware? Check before you download.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-5">
        {/* ═══ Input Panel ═══ */}
        <div
          className="bg-bg-secondary border border-border rounded-xl overflow-hidden animate-slide-up"
          style={{ "--delay": "60ms" } as React.CSSProperties}
        >
          {/* Device hardware card */}
          <div className="px-5 py-4 border-b border-border bg-bg-card/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${bestDevice ? "bg-success/10 border border-success/20" : "bg-warning/10 border border-warning/20"}`}>
                  <svg viewBox="0 0 24 24" fill="none" stroke={bestDevice ? "var(--color-success)" : "var(--color-warning)"} strokeWidth="1.5" className="w-5 h-5">
                    <rect x="4" y="4" width="16" height="16" rx="2" />
                    <path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3" strokeLinecap="round" />
                  </svg>
                </div>
                <div>
                  <div className="font-display font-semibold text-sm text-text-primary">{deviceName}</div>
                  <div className="text-[10px] font-mono text-text-muted">
                    {bestDevice?.device_type?.toUpperCase() || "UNKNOWN"} {bestDevice?.device_string ? `· ${bestDevice.device_string}` : ""}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono font-semibold text-lg text-text-primary">{freeMemory.toFixed(1)}</div>
                <div className="text-[10px] font-mono text-text-muted">GB free</div>
              </div>
            </div>
            {/* Memory bar */}
            {totalMemory > 0 && (
              <div className="mt-3">
                <div className="w-full h-1.5 bg-bg-primary rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${usedPct > 90 ? "bg-danger" : usedPct > 70 ? "bg-warning" : "bg-success"}`}
                    style={{ width: `${usedPct}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-[9px] font-mono text-text-muted">{usedMemory.toFixed(1)} GB used</span>
                  <span className="text-[9px] font-mono text-text-muted">{totalMemory.toFixed(1)} GB total</span>
                </div>
              </div>
            )}
          </div>

          <div className="p-5 space-y-4">
            {/* Quick select */}
            {aliases.length > 0 && (
              <div>
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-2">
                  Quick Select
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {aliases.slice(0, 8).map((a) => (
                    <button
                      key={a.alias}
                      onClick={() => selectAlias(a.alias)}
                      className={`group px-2.5 py-1.5 text-[11px] rounded-lg border transition-all cursor-pointer ${
                        modelId === a.repo_id
                          ? "bg-accent/10 border-accent/30 text-accent"
                          : "bg-bg-card border-border text-text-secondary hover:text-text-primary hover:border-accent/20"
                      }`}
                    >
                      <span className="font-medium">{a.display_name}</span>
                      <span className="ml-1.5 text-[9px] text-text-muted group-hover:text-text-secondary">
                        {a.param_count_billions}B
                      </span>
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
                onKeyDown={(e) => e.key === "Enter" && handleCheck()}
                placeholder="e.g., TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
                className="w-full bg-bg-input border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted/50 font-mono focus:outline-none focus:border-accent/50 transition-colors"
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
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-mono focus:outline-none focus:border-accent/50 cursor-pointer"
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
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-mono focus:outline-none focus:border-accent/50"
                />
              </div>
            </div>

            <button
              onClick={handleCheck}
              disabled={!modelId.trim() || isChecking}
              className="w-full py-2.5 rounded-xl bg-accent/15 border border-accent/30 text-accent text-sm font-display font-semibold hover:bg-accent/25 transition-all cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
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

        {/* ═══ Result Panel ═══ */}
        <div
          className="bg-bg-secondary border border-border rounded-xl overflow-hidden animate-slide-up"
          style={{ "--delay": "120ms" } as React.CSSProperties}
        >
          <div className="px-5 py-3 border-b border-border">
            <span className="text-xs text-text-secondary font-display font-semibold">Fit Report</span>
          </div>

          <div className="p-5">
            {error && (
              <div className="bg-danger-dim border border-danger/30 rounded-lg px-4 py-3 mb-4">
                <p className="text-sm text-danger">{error}</p>
              </div>
            )}

            {!result && !error && !isChecking && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <svg viewBox="0 0 48 48" fill="none" className="w-14 h-14 mb-4 text-text-muted/20">
                  <circle cx="24" cy="24" r="20" stroke="currentColor" strokeWidth="1.5" />
                  <circle cx="24" cy="24" r="8" stroke="currentColor" strokeWidth="1.5" />
                  <path d="M24 4v6M24 38v6M4 24h6M38 24h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                <p className="text-text-muted text-sm font-display">Select a model to check if it fits</p>
                <p className="text-text-muted/60 text-xs mt-1">No other inference tool does this</p>
              </div>
            )}

            {isChecking && (
              <div className="flex flex-col items-center justify-center py-16">
                <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mb-3" />
                <p className="text-text-muted text-sm font-display">Querying HuggingFace...</p>
              </div>
            )}

            {result && (
              <div className="space-y-5">
                {/* ── Verdict ── */}
                <div
                  className={`relative rounded-xl px-6 py-6 text-center border overflow-hidden animate-verdict ${
                    result.fits
                      ? "border-success/30"
                      : "border-danger/30"
                  }`}
                  style={{
                    background: result.fits
                      ? "linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.02))"
                      : "linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.02))",
                  }}
                >
                  {/* Animated check/x icon */}
                  <div className="flex justify-center mb-2">
                    {result.fits ? (
                      <svg viewBox="0 0 32 32" className="w-10 h-10">
                        <circle cx="16" cy="16" r="14" fill="none" stroke="var(--color-success)" strokeWidth="1.5" opacity="0.3" />
                        <path
                          d="M10 16l4 4 8-8"
                          fill="none"
                          stroke="var(--color-success)"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="animate-draw-check"
                        />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 32 32" className="w-10 h-10">
                        <circle cx="16" cy="16" r="14" fill="none" stroke="var(--color-danger)" strokeWidth="1.5" opacity="0.3" />
                        <path
                          d="M11 11l10 10M21 11l-10 10"
                          fill="none"
                          stroke="var(--color-danger)"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          className="animate-draw-x"
                        />
                      </svg>
                    )}
                  </div>
                  <p className={`font-display font-extrabold text-2xl ${result.fits ? "text-success" : "text-danger"}`}>
                    {result.fits ? "FITS" : "DOES NOT FIT"}
                  </p>
                  <p className="text-xs text-text-muted mt-1 font-mono">{result.device_name}</p>
                </div>

                {/* ── Visual Memory Breakdown ── */}
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-text-muted font-semibold mb-3">
                    Memory Breakdown
                  </p>

                  {/* Stacked bar visualization */}
                  {result.available_memory_gb != null && result.model_size_gb != null && (
                    <div className="mb-4">
                      <div className="w-full h-6 bg-bg-primary rounded-lg overflow-hidden flex">
                        {/* Model size segment */}
                        <div
                          className="h-full bg-accent/60 flex items-center justify-center transition-all"
                          style={{
                            width: `${Math.min((result.model_size_gb / result.available_memory_gb) * 100, 100)}%`,
                            minWidth: result.model_size_gb > 0 ? "2rem" : "0",
                          }}
                        >
                          <span className="text-[9px] font-mono text-white/80 truncate px-1">{result.model_size_gb} GB</span>
                        </div>
                        {/* Overhead segment */}
                        {(result.overhead_gb ?? 0) > 0 && (
                          <div
                            className="h-full bg-accent/30 flex items-center justify-center transition-all"
                            style={{
                              width: `${Math.min(((result.overhead_gb ?? 0) / result.available_memory_gb) * 100, 50)}%`,
                              minWidth: "1.5rem",
                            }}
                          >
                            <span className="text-[9px] font-mono text-white/50 truncate px-1">{result.overhead_gb} GB</span>
                          </div>
                        )}
                        {/* Headroom */}
                        {(result.headroom_gb ?? 0) > 0 && (
                          <div className="h-full flex-1 bg-success/15 flex items-center justify-center">
                            <span className="text-[9px] font-mono text-success/70 truncate px-1">{result.headroom_gb} GB free</span>
                          </div>
                        )}
                      </div>
                      <div className="flex justify-between mt-1">
                        <span className="text-[9px] font-mono text-accent/60">Model + Overhead</span>
                        <span className="text-[9px] font-mono text-text-muted">{result.available_memory_gb} GB available</span>
                      </div>
                    </div>
                  )}

                  {/* Numeric breakdown */}
                  <div className="space-y-1">
                    {[
                      { label: "Model Size", value: result.model_size_gb, color: "text-accent" },
                      { label: "Overhead (KV cache + runtime)", value: result.overhead_gb, color: "text-accent/60" },
                      { label: "Available Memory", value: result.available_memory_gb, color: "text-text-primary" },
                      {
                        label: "Headroom",
                        value: result.headroom_gb,
                        color: (result.headroom_gb ?? 0) >= 0 ? "text-success" : "text-danger",
                      },
                    ].map((row) => (
                      <div key={row.label} className="flex items-center justify-between bg-bg-card rounded-lg px-3 py-2">
                        <span className="text-xs text-text-secondary">{row.label}</span>
                        <span className={`text-sm font-mono font-medium ${row.color}`}>
                          {row.value != null ? `${row.value} GB` : "—"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ── Warnings ── */}
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

                {/* ── Alternatives Table ── */}
                {result.alternatives.length > 0 && (
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-semibold mb-2">
                      Quantization Alternatives
                    </p>
                    <div className="border border-border rounded-lg overflow-hidden">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-bg-card/50">
                            <th className="text-left px-3 py-2 text-text-muted font-medium font-mono">Quant</th>
                            <th className="text-right px-3 py-2 text-text-muted font-medium font-mono">Est. Size</th>
                            <th className="text-center px-3 py-2 text-text-muted font-medium">Fits?</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.alternatives.map((alt) => {
                            const isRec = alt.quantization === result.recommended_quantization;
                            return (
                              <tr
                                key={alt.quantization}
                                className={`border-t border-border transition-colors hover:bg-bg-card/30 ${
                                  isRec ? "bg-accent/5" : ""
                                }`}
                              >
                                <td className={`px-3 py-2 font-mono text-text-primary ${isRec ? "border-l-2 border-l-accent" : ""}`}>
                                  {alt.quantization}
                                  {isRec && (
                                    <span className="ml-2 text-[9px] px-1.5 py-0.5 rounded bg-accent/15 text-accent font-semibold uppercase">
                                      recommended
                                    </span>
                                  )}
                                </td>
                                <td className="text-right px-3 py-2 font-mono text-text-secondary">
                                  {alt.estimated_size_gb} GB
                                </td>
                                <td className="text-center px-3 py-2">
                                  <span className={`inline-block w-2.5 h-2.5 rounded-full ${alt.fits ? "bg-success" : "bg-danger"}`} />
                                </td>
                              </tr>
                            );
                          })}
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
