import { useDevices } from "../api/hooks";
import { formatGB } from "../utils/format";

const DEVICE_STYLES: Record<string, { color: string; bg: string; border: string; glow: string }> = {
  cuda:   { color: "text-green-400",  bg: "bg-green-400/10",  border: "border-green-400/30",  glow: "rgba(74,222,128,0.15)" },
  rocm:   { color: "text-red-400",    bg: "bg-red-400/10",    border: "border-red-400/30",    glow: "rgba(248,113,113,0.15)" },
  vulkan: { color: "text-yellow-400", bg: "bg-yellow-400/10", border: "border-yellow-400/30", glow: "rgba(250,204,21,0.15)" },
  cpu:    { color: "text-slate-400",  bg: "bg-slate-400/10",  border: "border-slate-400/30",  glow: "rgba(148,163,184,0.1)" },
};

const CHAIN_ORDER = ["cuda", "rocm", "vulkan", "cpu"];

export function DevicesPage() {
  const { data: devices, isLoading } = useDevices();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted">
        <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Sort devices into chain order
  const sortedDevices = [...(devices || [])].sort((a, b) => {
    const ai = CHAIN_ORDER.indexOf(a.device_type);
    const bi = CHAIN_ORDER.indexOf(b.device_type);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  return (
    <div className="space-y-5">
      {/* Chain explanation header */}
      <div className="animate-slide-up" style={{ "--delay": "0ms" } as React.CSSProperties}>
        <div className="flex items-center gap-3 mb-1">
          <svg viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" strokeWidth="1.5" className="w-6 h-6">
            <rect x="4" y="4" width="16" height="16" rx="2" />
            <path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3" strokeLinecap="round" />
          </svg>
          <h2 className="font-display font-bold text-xl text-text-primary tracking-tight">Hardware Devices</h2>
        </div>
        <p className="text-sm text-text-muted ml-9">
          UniInfer automatically falls back through your hardware chain if a device fails.
        </p>
      </div>

      {/* Fallback chain overview strip */}
      <div
        className="rounded-xl border border-border bg-bg-card px-5 py-3 animate-slide-up"
        style={{ "--delay": "60ms" } as React.CSSProperties}
      >
        <div className="text-[10px] font-mono text-text-muted uppercase tracking-widest mb-2">
          Fallback Priority
        </div>
        <div className="flex items-center gap-1">
          {CHAIN_ORDER.map((type, i) => {
            const style = DEVICE_STYLES[type];
            const hasDevice = sortedDevices?.some((d) => d.device_type === type);
            const isActive = sortedDevices?.some((d) => d.device_type === type && d.is_active);

            return (
              <div key={type} className="flex items-center">
                <div
                  className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium border transition-all ${
                    isActive
                      ? `${style.bg} ${style.border} ${style.color}`
                      : hasDevice
                      ? `bg-bg-secondary border-border ${style.color}/60`
                      : "bg-bg-secondary border-border text-text-muted/30"
                  }`}
                  style={isActive ? { boxShadow: `0 0 12px ${style.glow}` } : undefined}
                >
                  {type.toUpperCase()}
                  {isActive && <span className="ml-1.5 text-[9px] opacity-60">active</span>}
                  {!hasDevice && <span className="ml-1.5 text-[9px] opacity-40">n/a</span>}
                </div>
                {i < CHAIN_ORDER.length - 1 && (
                  <svg viewBox="0 0 16 16" className="w-4 h-4 text-text-muted/20 mx-0.5 shrink-0">
                    <path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Device nodes */}
      {!sortedDevices?.length ? (
        <div className="rounded-xl border border-border bg-bg-card p-8 text-center animate-slide-up" style={{ "--delay": "120ms" } as React.CSSProperties}>
          <svg viewBox="0 0 48 48" fill="none" className="w-14 h-14 mx-auto mb-4 text-text-muted/20">
            <rect x="8" y="8" width="32" height="32" rx="4" stroke="currentColor" strokeWidth="1.5" />
            <path d="M18 4v4M30 4v4M18 40v4M30 40v4M4 18h4M4 30h4M40 18h4M40 30h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M20 20l8 8M28 20l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <p className="font-display font-semibold text-text-secondary">No devices detected</p>
          <p className="text-xs text-text-muted mt-1">Check GPU drivers or verify CUDA/ROCm/Vulkan installation</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sortedDevices.map((d, i) => {
            const usedGb = d.total_memory_gb - d.free_memory_gb;
            const usedPct = d.total_memory_gb > 0 ? (usedGb / d.total_memory_gb) * 100 : 0;
            const style = DEVICE_STYLES[d.device_type] || DEVICE_STYLES.cpu;
            const memColor = usedPct > 90 ? "bg-danger" : usedPct > 70 ? "bg-warning" : "bg-success";

            return (
              <div
                key={d.device_string}
                className={`rounded-xl border overflow-hidden transition-all card-glow animate-slide-up ${
                  d.is_active
                    ? `${style.border} bg-bg-card`
                    : "border-border bg-bg-secondary"
                }`}
                style={{
                  "--delay": `${120 + i * 60}ms`,
                  ...(d.is_active ? { boxShadow: `0 0 20px ${style.glow}` } : {}),
                } as React.CSSProperties}
              >
                {/* Device header */}
                <div className="px-5 py-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      {/* Device type badge */}
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${style.bg} border ${style.border}`}>
                        <span className={`font-mono font-bold text-xs ${style.color}`}>
                          {d.device_type === "cuda" ? "CU" :
                           d.device_type === "rocm" ? "RC" :
                           d.device_type === "vulkan" ? "VK" : "CP"}
                        </span>
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-display font-semibold text-sm text-text-primary">
                            {d.name}
                          </span>
                          {d.is_active && (
                            <span className={`text-[9px] px-1.5 py-0.5 rounded ${style.bg} ${style.color} font-mono font-semibold uppercase`}>
                              Active
                            </span>
                          )}
                        </div>
                        <span className="text-[11px] text-text-muted font-mono">
                          {d.device_string}
                        </span>
                      </div>
                    </div>

                    {/* Memory focal point */}
                    <div className="text-right">
                      <div className="font-mono font-bold text-xl text-text-primary tabular-nums">
                        {d.free_memory_gb.toFixed(1)}
                      </div>
                      <div className="text-[10px] font-mono text-text-muted">GB free</div>
                    </div>
                  </div>
                </div>

                {/* Memory bar */}
                <div className="px-5 pb-4">
                  <div className="w-full h-2 bg-bg-primary rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${memColor}`}
                      style={{ width: `${usedPct}%` }}
                    />
                  </div>
                  <div className="flex justify-between mt-1.5 text-[10px] font-mono text-text-muted">
                    <span>{usedPct.toFixed(0)}% used ({formatGB(usedGb)})</span>
                    <span>{formatGB(d.total_memory_gb)} total</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
