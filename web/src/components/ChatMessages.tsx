import { useEffect, useRef } from "react";
import type { ChatMessage } from "../api/types";

interface Props {
  messages: ChatMessage[];
  model: string;
  source: string;
  sessionId: string;
}

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function ChatMessages({ messages, model, source, sessionId }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm">
        No messages in this session
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Session header */}
      <div className="px-5 py-3 border-b border-border bg-bg-secondary/50 flex items-center gap-3 shrink-0">
        <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
        <span className="text-sm font-medium text-text-primary font-[JetBrains_Mono]">
          {model || "unknown"}
        </span>
        <span className="text-[10px] text-text-muted uppercase tracking-wider">
          via {source}
        </span>
        <span className="ml-auto text-[10px] text-text-muted font-[JetBrains_Mono]">
          {sessionId.slice(0, 8)}
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {messages.map((msg, i) => {
          const isUser = msg.role === "user";
          const isSystem = msg.role === "system";

          if (isSystem) {
            return (
              <div key={i} className="flex justify-center animate-fade-in" style={{ animationDelay: `${i * 30}ms` }}>
                <div className="bg-warning-dim border border-warning/20 rounded-lg px-4 py-2 max-w-lg">
                  <span className="text-[10px] uppercase tracking-wider text-warning font-semibold block mb-0.5">
                    system
                  </span>
                  <p className="text-xs text-text-secondary whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                  </p>
                </div>
              </div>
            );
          }

          return (
            <div
              key={i}
              className={`flex animate-fade-in ${isUser ? "justify-end" : "justify-start"}`}
              style={{ animationDelay: `${i * 30}ms` }}
            >
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                  isUser
                    ? "bg-accent/15 border border-accent/20 rounded-br-md"
                    : "bg-bg-card border border-border rounded-bl-md"
                }`}
              >
                <p className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed break-words">
                  {msg.content}
                </p>

                <div className="flex items-center gap-2 mt-2">
                  <span className="text-[10px] text-text-muted">
                    {formatTimestamp(msg.timestamp)}
                  </span>
                  {!isUser && msg.tokens > 0 && (
                    <>
                      <span className="text-[10px] text-text-muted">·</span>
                      <span className="text-[10px] text-text-muted font-[JetBrains_Mono]">
                        {msg.tokens} tok
                      </span>
                      {msg.tokens_per_second > 0 && (
                        <>
                          <span className="text-[10px] text-text-muted">·</span>
                          <span className="text-[10px] text-accent font-[JetBrains_Mono]">
                            {msg.tokens_per_second.toFixed(1)} tok/s
                          </span>
                        </>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
