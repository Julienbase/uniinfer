import { useStatus } from "../api/hooks";
import { useSSE } from "../hooks/useSSE";
import { StatusCard } from "../components/StatusCard";
import { MemoryGauge } from "../components/MemoryGauge";
import { ThroughputChart } from "../components/ThroughputChart";
import { DiagnosticsPanel } from "../components/DiagnosticsPanel";
import { QueueIndicator } from "../components/QueueIndicator";

export function DashboardPage() {
  const { data: status, isLoading, error } = useStatus();
  const { connected, tokHistory } = useSSE();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Connecting to server...</span>
        </div>
      </div>
    );
  }

  if (error || !status) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="bg-danger-dim border border-danger/30 rounded-xl p-6 max-w-md text-center">
          <p className="text-danger font-semibold mb-1">Connection Failed</p>
          <p className="text-sm text-text-secondary">
            Cannot reach the UniInfer server. Make sure it's running with{" "}
            <code className="text-xs bg-bg-card px-1.5 py-0.5 rounded font-[JetBrains_Mono]">
              uniinfer serve
            </code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Top row: Status + Memory + Queue */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatusCard status={status} connected={connected} />
        <MemoryGauge
          totalGb={status.device_memory_total_gb}
          freeGb={status.device_memory_free_gb}
          fitSizeGb={status.fit?.model_size_gb}
        />
        <QueueIndicator
          queueDepth={status.scheduler.queue_depth}
          isProcessing={status.scheduler.is_processing}
        />
      </div>

      {/* Throughput chart */}
      <ThroughputChart
        history={tokHistory}
        currentTokS={status.diagnostics.average_tokens_per_second}
        peakTokS={status.diagnostics.peak_tokens_per_second}
      />

      {/* Diagnostics */}
      <DiagnosticsPanel diagnostics={status.diagnostics} />
    </div>
  );
}
