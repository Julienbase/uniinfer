import type { ChatSessionSummary } from "../api/types";

interface Props {
  sessions: ChatSessionSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const sourceColors: Record<string, string> = {
  cli: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  api: "bg-blue-400/20 text-blue-300 border-blue-400/30",
  dashboard: "bg-purple-400/20 text-purple-300 border-purple-400/30",
};

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const now = new Date();
  const diff = now.getTime() - d.getTime();

  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function ChatSessionList({ sessions, selectedId, onSelect }: Props) {
  if (sessions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-12 px-4 text-center">
        <div className="w-10 h-10 rounded-xl bg-bg-card border border-border flex items-center justify-center mb-3">
          <svg className="w-5 h-5 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 01-.825-.242m9.345-8.334a2.126 2.126 0 00-.476-.095 48.64 48.64 0 00-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0011.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155" />
          </svg>
        </div>
        <p className="text-text-muted text-sm">No chat sessions yet</p>
        <p className="text-text-muted/60 text-xs mt-1">
          Start chatting via CLI or API
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-1 p-2">
      {sessions.map((s) => {
        const isSelected = s.session_id === selectedId;
        const colorClass = sourceColors[s.source] || sourceColors.api;

        return (
          <button
            key={s.session_id}
            onClick={() => onSelect(s.session_id)}
            className={`w-full text-left px-3 py-2.5 rounded-lg transition-all cursor-pointer group ${
              isSelected
                ? "bg-accent/12 border border-accent/25"
                : "hover:bg-bg-card-hover border border-transparent"
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span
                className={`text-[10px] font-[JetBrains_Mono] uppercase px-1.5 py-0.5 rounded border ${colorClass}`}
              >
                {s.source}
              </span>
              <span className="text-[10px] text-text-muted">
                {formatTime(s.created_at)}
              </span>
            </div>

            <p className="text-xs text-text-secondary font-medium truncate">
              {s.model || "unknown"}
            </p>

            {s.last_message_preview && (
              <p className="text-[11px] text-text-muted mt-0.5 truncate leading-relaxed">
                {s.last_message_preview}
              </p>
            )}

            <div className="flex items-center gap-1.5 mt-1.5">
              <span className="text-[10px] text-text-muted font-[JetBrains_Mono]">
                {s.message_count} msg{s.message_count !== 1 ? "s" : ""}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
