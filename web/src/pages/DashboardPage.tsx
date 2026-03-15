import { useStatus } from "../api/hooks";
import { useSSE } from "../hooks/useSSE";
import { StatusCard } from "../components/StatusCard";
import { MemoryGauge } from "../components/MemoryGauge";
import { ThroughputChart } from "../components/ThroughputChart";
import { DiagnosticsPanel } from "../components/DiagnosticsPanel";
import { QueueIndicator } from "../components/QueueIndicator";

const DEVICE_CHAIN = [
  { type: "CUDA", color: "text-green-400", bg: "bg-green-400/10", border: "border-green-400/30" },
  { type: "ROCm", color: "text-red-400", bg: "bg-red-400/10", border: "border-red-400/30" },
  { type: "Vulkan", color: "text-yellow-400", bg: "bg-yellow-400/10", border: "border-yellow-400/30" },
  { type: "CPU", color: "text-slate-400", bg: "bg-slate-400/10", border: "border-slate-400/30" },
];

export function DashboardPage() {
  const { data: status, isLoading, error } = useStatus();
  const { connected, tokHistory } = useSSE();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-body">Connecting to server...</span>
        </div>
      </div>
    );
  }

  if (error || !status) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="bg-danger-dim border border-danger/30 rounded-xl p-6 max-w-md text-center">
          <p className="text-danger font-display font-semibold mb-1">Connection Failed</p>
          <p className="text-sm text-text-secondary">
            Cannot reach the UniInfer server. Make sure it&apos;s running with{" "}
            <code className="text-xs bg-bg-card px-1.5 py-0.5 rounded font-mono">
              uniinfer serve
            </code>
          </p>
        </div>
      </div>
    );
  }

  const modelLoaded = status.loaded;
  const activeBackend = (status.backend || "").toLowerCase();
  const currentTokS = status.diagnostics.average_tokens_per_second;
  const peakTokS = status.diagnostics.peak_tokens_per_second;
  const fallback = status.fallback;

  return (
    <div className="space-y-3">
      {/* ═══ Hero Banner ═══ */}
      <div
        className="relative rounded-2xl border border-border bg-bg-card overflow-hidden animate-slide-up"
        style={{ "--delay": "0ms" } as React.CSSProperties}
      >
        {/* Radial glow background */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: modelLoaded
              ? "radial-gradient(ellipse at 30% 50%, rgba(245,158,11,0.04) 0%, transparent 70%)"
              : "radial-gradient(ellipse at 50% 50%, rgba(100,100,120,0.04) 0%, transparent 70%)",
          }}
        />

        <div className="relative px-6 py-5">
          {modelLoaded ? (
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-success animate-pulse" />
                  <span className="text-xs font-mono text-text-muted uppercase tracking-widest">Active Model</span>
                </div>
                <h1 className="font-display font-extrabold text-2xl md:text-3xl text-text-primary tracking-tight">
                  {status.model || "Unknown"}
                </h1>
                <div className="flex flex-wrap gap-2 mt-1">
                  {[
                    { label: status.device || "—", icon: "device" },
                    { label: status.backend || "—", icon: "backend" },
                    { label: status.quantization ? status.quantization.toUpperCase() : "—", icon: "quant" },
                    { label: status.context_length ? `${status.context_length.toLocaleString()} ctx` : "—", icon: "ctx" },
                  ].map((chip) => (
                    <span
                      key={chip.icon}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-bg-secondary border border-border text-xs font-mono text-text-secondary"
                    >
                      {chip.label}
                    </span>
                  ))}
                </div>
              </div>

              {/* Hero metric — throughput */}
              <div className="text-right shrink-0 pl-6">
                <div className="text-xs font-mono text-text-muted uppercase tracking-widest mb-1">Throughput</div>
                <div className="font-display font-extrabold text-4xl text-accent tabular-nums">
                  {currentTokS > 0 ? currentTokS.toFixed(1) : "—"}
                </div>
                <div className="text-xs font-mono text-text-muted">tok/s avg</div>
                {peakTokS > 0 && (
                  <div className="text-xs font-mono text-success mt-1">
                    {peakTokS.toFixed(1)} peak
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* No model loaded — CTA */
            <div className="flex flex-col items-center py-6 text-center">
              <div className="w-16 h-16 rounded-2xl border-2 border-dashed border-accent/30 flex items-center justify-center mb-4">
                <svg viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" strokeWidth="1.5" className="w-8 h-8 opacity-50">
                  <path d="M12 4v16M4 12h16" strokeLinecap="round" />
                </svg>
              </div>
              <h2 className="font-display font-bold text-xl text-text-secondary mb-1">No Model Loaded</h2>
              <p className="text-sm text-text-muted max-w-sm">
                Go to <span className="text-accent font-medium">Models</span> to download and load a model, or start with{" "}
                <code className="font-mono text-xs bg-bg-secondary px-1.5 py-0.5 rounded">uniinfer serve &lt;model&gt;</code>
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ═══ Hardware Fallback Chain ═══ */}
      <div
        className="rounded-xl border border-border bg-bg-card px-5 py-3 animate-slide-up"
        style={{ "--delay": "60ms" } as React.CSSProperties}
      >
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-mono text-text-muted uppercase tracking-widest">Hardware Fallback Chain</span>
          {fallback?.fell_back && (
            <span className="text-[10px] px-2 py-0.5 rounded bg-warning-dim text-warning font-mono">
              Fallback active
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 mt-2">
          {DEVICE_CHAIN.map((d, i) => {
            const isActive = activeBackend.includes(d.type.toLowerCase()) ||
              (status.device || "").toLowerCase().includes(d.type.toLowerCase());
            return (
              <div key={d.type} className="flex items-center">
                <div
                  className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium border transition-all ${
                    isActive
                      ? `${d.bg} ${d.border} ${d.color} shadow-sm`
                      : "bg-bg-secondary border-border text-text-muted/40"
                  }`}
                  style={isActive ? { boxShadow: "0 0 12px rgba(245,158,11,0.08)" } : undefined}
                >
                  {d.type}
                </div>
                {i < DEVICE_CHAIN.length - 1 && (
                  <svg viewBox="0 0 16 16" className="w-4 h-4 text-text-muted/30 mx-0.5 shrink-0">
                    <path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ═══ System Grid ═══ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div
          className="animate-slide-up card-glow"
          style={{ "--delay": "120ms" } as React.CSSProperties}
        >
          <StatusCard status={status} connected={connected} />
        </div>
        <div
          className="animate-slide-up card-glow"
          style={{ "--delay": "180ms" } as React.CSSProperties}
        >
          <MemoryGauge
            totalGb={status.device_memory_total_gb}
            freeGb={status.device_memory_free_gb}
            fitSizeGb={status.fit?.model_size_gb}
          />
        </div>
      </div>

      {/* ═══ Throughput Chart ═══ */}
      <div
        className="scan-line animate-slide-up card-glow"
        style={{ "--delay": "240ms" } as React.CSSProperties}
      >
        <ThroughputChart
          history={tokHistory}
          currentTokS={currentTokS}
          peakTokS={peakTokS}
        />
      </div>

      {/* ═══ Bottom Row: Diagnostics + Queue ═══ */}
      <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-4">
        <div
          className="animate-slide-up card-glow"
          style={{ "--delay": "300ms" } as React.CSSProperties}
        >
          <DiagnosticsPanel diagnostics={status.diagnostics} />
        </div>
        <div
          className="animate-slide-up card-glow"
          style={{ "--delay": "360ms" } as React.CSSProperties}
        >
          <QueueIndicator
            queueDepth={status.scheduler.queue_depth}
            isProcessing={status.scheduler.is_processing}
          />
        </div>
      </div>
    </div>
  );
}
