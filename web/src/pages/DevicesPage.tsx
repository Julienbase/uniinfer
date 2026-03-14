import { useDevices } from "../api/hooks";
import { formatGB } from "../utils/format";

export function DevicesPage() {
  const { data: devices, isLoading } = useDevices();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted">
        <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary mb-4">
          Hardware Devices
        </h2>

        {!devices?.length ? (
          <div className="text-text-muted text-sm py-4 text-center">
            No devices detected.
          </div>
        ) : (
          <div className="space-y-3">
            {devices.map((d) => {
              const usedGb = d.total_memory_gb - d.free_memory_gb;
              const usedPct = d.total_memory_gb > 0 ? (usedGb / d.total_memory_gb) * 100 : 0;

              return (
                <div
                  key={d.device_string}
                  className={`p-4 rounded-lg border transition-colors ${
                    d.is_active
                      ? "bg-accent-glow border-accent/30"
                      : "bg-bg-secondary border-border hover:border-border-bright"
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-bg-card flex items-center justify-center text-sm">
                        {d.device_type === "cuda" ? "🟢" :
                         d.device_type === "rocm" ? "🔴" :
                         d.device_type === "vulkan" ? "🟡" : "⚪"}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-text-primary">
                            {d.name}
                          </span>
                          {d.is_active && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/20 text-accent font-semibold uppercase">
                              Active
                            </span>
                          )}
                        </div>
                        <span className="text-xs text-text-muted font-[JetBrains_Mono]">
                          {d.device_string} ({d.device_type})
                        </span>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-sm font-[JetBrains_Mono] text-text-primary">
                        {formatGB(d.total_memory_gb)}
                      </div>
                      <div className="text-xs text-text-muted">
                        {formatGB(d.free_memory_gb)} free
                      </div>
                    </div>
                  </div>

                  {/* Memory bar */}
                  <div className="w-full h-1.5 bg-bg-card rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        usedPct > 90 ? "bg-danger" :
                        usedPct > 70 ? "bg-warning" :
                        "bg-accent"
                      }`}
                      style={{ width: `${usedPct}%` }}
                    />
                  </div>
                  <div className="flex justify-between mt-1 text-[10px] text-text-muted">
                    <span>{usedPct.toFixed(0)}% used</span>
                    <span>{formatGB(usedGb)} / {formatGB(d.total_memory_gb)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
