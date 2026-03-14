interface Props {
  queueDepth: number;
  isProcessing: boolean;
}

export function QueueIndicator({ queueDepth, isProcessing }: Props) {
  return (
    <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
      <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary mb-4">
        Queue
      </h2>

      <div className="flex items-center gap-6">
        <div className="flex flex-col items-center">
          <span className="text-3xl font-[JetBrains_Mono] font-bold text-text-primary">
            {queueDepth}
          </span>
          <span className="text-[10px] uppercase tracking-widest text-text-muted mt-1">
            Pending
          </span>
        </div>

        <div className="flex flex-col items-center">
          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isProcessing ? "bg-accent" : "bg-border-bright"
              }`}
              style={isProcessing ? {
                animation: "pulse-glow 1.5s infinite",
                boxShadow: "0 0 8px rgba(129, 140, 248, 0.5)",
              } : {}}
            />
            <span className="text-sm text-text-secondary">
              {isProcessing ? "Processing" : "Idle"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
