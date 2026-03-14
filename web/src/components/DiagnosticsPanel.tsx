import type { DiagnosticsInfo } from "../api/types";
import { formatNumber, formatDuration, formatTokS } from "../utils/format";

interface Props {
  diagnostics: DiagnosticsInfo;
}

export function DiagnosticsPanel({ diagnostics }: Props) {
  const stats = [
    {
      label: "Total Inferences",
      value: formatNumber(diagnostics.total_inferences),
      sub: null,
    },
    {
      label: "Tokens Generated",
      value: formatNumber(diagnostics.total_tokens_generated),
      sub: null,
    },
    {
      label: "Avg Throughput",
      value: formatTokS(diagnostics.average_tokens_per_second),
      sub: null,
    },
    {
      label: "Peak Throughput",
      value: formatTokS(diagnostics.peak_tokens_per_second),
      sub: null,
    },
    {
      label: "Inference Time",
      value: formatDuration(diagnostics.total_inference_time_seconds),
      sub: null,
    },
    {
      label: "Model Load Time",
      value: diagnostics.model_load_time_seconds > 0
        ? `${diagnostics.model_load_time_seconds.toFixed(2)}s`
        : "—",
      sub: null,
    },
  ];

  return (
    <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
      <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary mb-4">
        Session Diagnostics
      </h2>

      <div className="grid grid-cols-3 gap-4">
        {stats.map((s) => (
          <div key={s.label} className="flex flex-col">
            <span className="text-[10px] uppercase tracking-widest text-text-muted mb-1">
              {s.label}
            </span>
            <span className="text-lg font-[JetBrains_Mono] font-semibold text-text-primary">
              {s.value}
            </span>
          </div>
        ))}
      </div>

      {diagnostics.last_inference && (
        <div className="mt-4 pt-3 border-t border-border">
          <span className="text-[10px] uppercase tracking-widest text-text-muted">
            Last Inference
          </span>
          <div className="mt-1.5 flex gap-4 text-xs text-text-secondary font-[JetBrains_Mono]">
            <span>{diagnostics.last_inference.method}</span>
            <span>{diagnostics.last_inference.total_tokens} tok</span>
            <span>{diagnostics.last_inference.elapsed_seconds.toFixed(2)}s</span>
            <span className="text-accent">
              {formatTokS(diagnostics.last_inference.tokens_per_second)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
