import type { StatusResponse } from "../api/types";
import { formatDuration } from "../utils/format";

interface Props {
  status: StatusResponse;
  connected: boolean;
}

export function StatusCard({ status, connected }: Props) {
  const items = [
    { label: "Model", value: status.model || "—", mono: true },
    { label: "Device", value: status.device_name || "—" },
    { label: "Backend", value: status.backend || "—" },
    { label: "Quantization", value: status.quantization?.toUpperCase() || "—", mono: true },
    { label: "Context", value: status.context_length ? `${status.context_length.toLocaleString()} tokens` : "—" },
    { label: "Uptime", value: formatDuration(status.uptime_seconds) },
  ];

  return (
    <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary">
          System Status
        </h2>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              status.loaded
                ? connected
                  ? "bg-success"
                  : "bg-warning"
                : "bg-danger"
            }`}
            style={{
              animation: status.loaded && connected ? "pulse-glow 2s infinite" : "none",
              boxShadow: status.loaded && connected
                ? "0 0 6px rgba(52, 211, 153, 0.5)"
                : "none",
            }}
          />
          <span className="text-xs text-text-muted">
            {status.loaded ? (connected ? "Live" : "Polling") : "Offline"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {items.map((item) => (
          <div key={item.label} className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-widest text-text-muted">
              {item.label}
            </span>
            <span
              className={`text-sm text-text-primary truncate ${
                item.mono ? 'font-[JetBrains_Mono]' : ''
              }`}
              title={item.value}
            >
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
