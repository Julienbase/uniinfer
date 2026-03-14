import { useCallback, useEffect, useRef, useState } from "react";
import { useChatSessions, useChatSession, useRecentMessages, useStatus, useCachedModels } from "../api/hooks";
import { sendChatMessage, loadModel } from "../api/client";
import { ChatSessionList } from "../components/ChatSessionList";
import { ChatMessages } from "../components/ChatMessages";
import type { ChatMessage } from "../api/types";
import { useQueryClient } from "@tanstack/react-query";

type View = "playground" | "sessions" | "feed";

interface PlaygroundMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export function ChatPage() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [view, setView] = useState<View>("playground");

  const { data: sessions = [], isLoading: loadingSessions } = useChatSessions();
  const { data: session } = useChatSession(selectedSessionId);
  const { data: recentMessages = [] } = useRecentMessages(80);
  const { data: status } = useStatus();
  const { data: cachedModels } = useCachedModels();
  const queryClient = useQueryClient();

  const messages: ChatMessage[] = session?.messages ?? [];

  // Model switching state
  const [switchingModel, setSwitchingModel] = useState(false);
  const [switchError, setSwitchError] = useState<string | null>(null);

  // Playground state
  const [playgroundMessages, setPlaygroundMessages] = useState<PlaygroundMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [showSettings, setShowSettings] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const modelName = status?.model || "model";
  const modelLoaded = status?.loaded ?? false;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [playgroundMessages.length, streamingText]);

  const handleSend = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    const userMsg: PlaygroundMessage = { role: "user", content: trimmed };
    const updatedMessages = [...playgroundMessages, userMsg];
    setPlaygroundMessages(updatedMessages);
    setInput("");
    setIsStreaming(true);
    setStreamingText("");

    const controller = new AbortController();
    abortRef.current = controller;

    let accumulated = "";
    try {
      await sendChatMessage(
        {
          messages: updatedMessages.map((m) => ({ role: m.role, content: m.content })),
          session_id: sessionId,
          temperature,
          max_tokens: maxTokens,
          system_prompt: systemPrompt || undefined,
        },
        (token) => {
          accumulated += token;
          setStreamingText(accumulated);
        },
        (returnedSessionId) => {
          if (returnedSessionId) setSessionId(returnedSessionId);
          setPlaygroundMessages((prev) => [
            ...prev,
            { role: "assistant", content: accumulated },
          ]);
          setStreamingText("");
          setIsStreaming(false);
        },
        controller.signal,
      );
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        if (accumulated.length > 0) {
          setPlaygroundMessages((prev) => [
            ...prev,
            { role: "assistant", content: accumulated + " [stopped]" },
          ]);
        }
      } else {
        const errorMsg = err instanceof Error ? err.message : "Unknown error";
        setPlaygroundMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${errorMsg}` },
        ]);
      }
      setStreamingText("");
      setIsStreaming(false);
    }
  }, [input, isStreaming, playgroundMessages, sessionId, temperature, maxTokens, systemPrompt]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleNewChat = useCallback(() => {
    setPlaygroundMessages([]);
    setSessionId(undefined);
    setStreamingText("");
    setIsStreaming(false);
    inputRef.current?.focus();
  }, []);

  async function handleSwitchModel(modelId: string, quantization: string) {
    setSwitchingModel(true);
    setSwitchError(null);
    try {
      const result = await loadModel({ model_id: modelId, quantization });
      if (!result.success) {
        setSwitchError(result.error || "Failed to switch model");
      } else {
        queryClient.invalidateQueries({ queryKey: ["status"] });
        queryClient.invalidateQueries({ queryKey: ["cached-models"] });
      }
    } catch (err) {
      setSwitchError(err instanceof Error ? err.message : "Switch failed");
    } finally {
      setSwitchingModel(false);
    }
  }

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="space-y-4">
      {/* View toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-text-primary tracking-tight">Chat</h2>
          {sessions.length > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-card border border-border text-text-muted font-[JetBrains_Mono]">
              {sessions.length} session{sessions.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        <div className="flex gap-1 bg-bg-card rounded-lg p-0.5 border border-border">
          {(["playground", "sessions", "feed"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1 text-xs rounded-md transition-all cursor-pointer capitalize ${
                view === v
                  ? "bg-accent/15 text-accent font-medium"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Model switch error */}
      {switchError && (
        <div className="bg-danger-dim border border-danger/30 rounded-xl p-3 animate-fade-in flex items-center justify-between">
          <p className="text-sm text-danger">{switchError}</p>
          <button onClick={() => setSwitchError(null)} className="text-danger/60 hover:text-danger text-sm cursor-pointer">
            Dismiss
          </button>
        </div>
      )}

      {/* ============= PLAYGROUND VIEW ============= */}
      {view === "playground" && (
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden flex flex-col" style={{ height: "560px" }}>
          {/* Top bar */}
          <div className="px-4 py-2.5 border-b border-border bg-bg-secondary/80 flex items-center gap-3 shrink-0">
            <div className={`w-2 h-2 rounded-full ${modelLoaded ? "bg-success animate-pulse" : "bg-danger"}`} />

            {/* Model selector dropdown */}
            {cachedModels && cachedModels.length > 1 ? (
              <div className="relative">
                <select
                  value={`${status?.model || ""}::${status?.quantization || ""}`}
                  onChange={(e) => {
                    const [id, quant] = e.target.value.split("::");
                    if (id && quant) handleSwitchModel(id, quant);
                  }}
                  disabled={switchingModel || isStreaming}
                  className="text-sm font-medium text-text-primary font-[JetBrains_Mono] bg-transparent border border-border rounded-md px-2 py-0.5 cursor-pointer focus:outline-none focus:border-accent/50 disabled:opacity-50 appearance-none pr-6"
                  style={{ backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E\")", backgroundRepeat: "no-repeat", backgroundPosition: "right 4px center" }}
                >
                  {cachedModels.map((m) => (
                    <option key={`${m.model_id}::${m.quantization}`} value={`${m.model_id}::${m.quantization}`}>
                      {m.model_id.split("/").pop()} ({m.quantization.toUpperCase()})
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <span className="text-sm font-medium text-text-primary font-[JetBrains_Mono]">
                {modelName}
              </span>
            )}

            {switchingModel && (
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                <span className="text-[10px] text-accent">Switching...</span>
              </div>
            )}

            {sessionId && (
              <span className="text-[10px] text-text-muted font-[JetBrains_Mono]">
                {sessionId.slice(0, 12)}
              </span>
            )}

            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-1.5 rounded-md transition-all cursor-pointer ${
                  showSettings ? "bg-accent/15 text-accent" : "text-text-muted hover:text-text-secondary hover:bg-bg-card"
                }`}
                title="Settings"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10.343 3.94c.09-.542.56-.94 1.11-.94h1.093c.55 0 1.02.398 1.11.94l.149.894c.07.424.384.764.78.93.398.164.855.142 1.205-.108l.737-.527a1.125 1.125 0 011.45.12l.773.774c.39.389.44 1.002.12 1.45l-.527.737c-.25.35-.272.806-.107 1.204.165.397.505.71.93.78l.893.15c.543.09.94.56.94 1.109v1.094c0 .55-.397 1.02-.94 1.11l-.893.149c-.425.07-.765.383-.93.78-.165.398-.143.854.107 1.204l.527.738c.32.447.269 1.06-.12 1.45l-.774.773a1.125 1.125 0 01-1.449.12l-.738-.527c-.35-.25-.806-.272-1.204-.107-.397.165-.71.505-.78.929l-.15.894c-.09.542-.56.94-1.11.94h-1.094c-.55 0-1.019-.398-1.11-.94l-.148-.894c-.071-.424-.384-.764-.781-.93-.398-.164-.854-.142-1.204.108l-.738.527c-.447.32-1.06.269-1.45-.12l-.773-.774a1.125 1.125 0 01-.12-1.45l.527-.737c.25-.35.273-.806.108-1.204-.165-.397-.506-.71-.93-.78l-.894-.15c-.542-.09-.94-.56-.94-1.109v-1.094c0-.55.398-1.02.94-1.11l.894-.149c.424-.07.765-.383.93-.78.165-.398.143-.854-.107-1.204l-.527-.738a1.125 1.125 0 01.12-1.45l.773-.773a1.125 1.125 0 011.45-.12l.737.527c.35.25.807.272 1.204.107.397-.165.71-.505.78-.929l.15-.894z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
              <button
                onClick={handleNewChat}
                className="px-2.5 py-1 text-[11px] rounded-md bg-bg-card border border-border text-text-secondary hover:text-text-primary hover:border-border-bright transition-all cursor-pointer"
              >
                New Chat
              </button>
            </div>
          </div>

          {/* Settings panel (collapsible) */}
          {showSettings && (
            <div className="px-4 py-3 border-b border-border bg-bg-card/50 space-y-3 shrink-0 animate-fade-in">
              <div>
                <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                  System Prompt
                </label>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="You are a helpful assistant..."
                  rows={2}
                  className="w-full bg-bg-input border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted/50 resize-none focus:outline-none focus:border-accent/50"
                />
              </div>
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                    Temperature: {temperature.toFixed(1)}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-accent h-1"
                  />
                </div>
                <div className="w-32">
                  <label className="text-[10px] uppercase tracking-wider text-text-muted font-semibold block mb-1">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={4096}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value) || 512)}
                    className="w-full bg-bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary font-[JetBrains_Mono] focus:outline-none focus:border-accent/50"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Messages area */}
          <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
            {playgroundMessages.length === 0 && !isStreaming && (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="w-16 h-16 rounded-2xl bg-accent/8 border border-accent/15 flex items-center justify-center mb-4">
                  <span className="text-2xl font-bold text-accent font-[JetBrains_Mono]">U</span>
                </div>
                <p className="text-text-secondary text-sm font-medium mb-1">
                  Chat with {modelName}
                </p>
                <p className="text-text-muted text-xs max-w-sm">
                  Send a message to start a conversation. Your chat will be tracked in the session history.
                </p>
              </div>
            )}

            {playgroundMessages.map((msg, i) => {
              const isUser = msg.role === "user";
              return (
                <div
                  key={i}
                  className={`flex animate-fade-in ${isUser ? "justify-end" : "justify-start"}`}
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
                  </div>
                </div>
              );
            })}

            {/* Streaming response */}
            {isStreaming && (
              <div className="flex justify-start animate-fade-in">
                <div className="max-w-[75%] rounded-2xl rounded-bl-md px-4 py-3 bg-bg-card border border-border">
                  {streamingText ? (
                    <p className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed break-words">
                      {streamingText}
                      <span className="inline-block w-2 h-4 bg-accent/60 ml-0.5 animate-pulse" />
                    </p>
                  ) : (
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                      <div className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                      <div className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                  )}
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input area */}
          <div className="px-4 py-3 border-t border-border bg-bg-secondary/80 shrink-0">
            {!modelLoaded ? (
              <div className="flex items-center justify-center py-2 text-sm text-danger">
                No model loaded. Start the server with a model first.
              </div>
            ) : (
              <div className="flex gap-2 items-end">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
                  rows={1}
                  disabled={isStreaming}
                  className="flex-1 bg-bg-input border border-border rounded-xl px-4 py-2.5 text-sm text-text-primary placeholder-text-muted/50 resize-none focus:outline-none focus:border-accent/50 disabled:opacity-50 max-h-32"
                  style={{ minHeight: "42px" }}
                  onInput={(e) => {
                    const target = e.target as HTMLTextAreaElement;
                    target.style.height = "42px";
                    target.style.height = Math.min(target.scrollHeight, 128) + "px";
                  }}
                />
                {isStreaming ? (
                  <button
                    onClick={handleStop}
                    className="px-4 py-2.5 rounded-xl bg-danger/15 border border-danger/30 text-danger text-sm font-medium hover:bg-danger/25 transition-all cursor-pointer shrink-0"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    onClick={handleSend}
                    disabled={!input.trim()}
                    className="px-4 py-2.5 rounded-xl bg-accent/15 border border-accent/30 text-accent text-sm font-medium hover:bg-accent/25 transition-all cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed shrink-0"
                  >
                    Send
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ============= SESSIONS VIEW ============= */}
      {view === "sessions" && (
        <div className="grid grid-cols-1 md:grid-cols-[280px_1fr] gap-4 min-h-[520px]">
          <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
            {loadingSessions ? (
              <div className="flex items-center justify-center h-64 text-text-muted">
                <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              </div>
            ) : (
              <div className="h-[520px] overflow-y-auto">
                <ChatSessionList
                  sessions={sessions}
                  selectedId={selectedSessionId}
                  onSelect={setSelectedSessionId}
                />
              </div>
            )}
          </div>

          <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
            {selectedSessionId && session ? (
              <ChatMessages
                messages={messages}
                model={session.model}
                source={session.source}
                sessionId={session.session_id}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full py-20 text-center">
                <div className="w-14 h-14 rounded-2xl bg-bg-card border border-border flex items-center justify-center mb-4">
                  <svg className="w-7 h-7 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 011.037-.443 48.282 48.282 0 005.68-.494c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
                  </svg>
                </div>
                <p className="text-text-secondary text-sm font-medium">Select a session</p>
                <p className="text-text-muted text-xs mt-1">
                  Choose a chat session to view the conversation
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ============= LIVE FEED VIEW ============= */}
      {view === "feed" && (
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden min-h-[520px]">
          {recentMessages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <div className="w-10 h-10 rounded-xl bg-bg-card border border-border flex items-center justify-center mb-3">
                <svg className="w-5 h-5 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                </svg>
              </div>
              <p className="text-text-muted text-sm">No messages yet</p>
              <p className="text-text-muted/60 text-xs mt-1">Messages will appear here in real-time</p>
            </div>
          ) : (
            <div className="divide-y divide-border">
              <div className="px-5 py-3 bg-bg-secondary/50 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
                <span className="text-xs text-text-secondary font-medium">
                  Live — {recentMessages.length} recent messages
                </span>
              </div>

              <div className="max-h-[470px] overflow-y-auto">
                {recentMessages.map((msg, i) => {
                  const isUser = msg.role === "user";
                  return (
                    <div
                      key={i}
                      className="px-5 py-3 hover:bg-bg-card/50 transition-colors animate-fade-in"
                      style={{ animationDelay: `${i * 20}ms` }}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className={`text-[10px] font-[JetBrains_Mono] font-semibold uppercase ${
                            isUser ? "text-accent" : "text-success"
                          }`}
                        >
                          {msg.role}
                        </span>
                        <span className="text-[10px] text-text-muted">·</span>
                        <span className="text-[10px] text-text-muted font-[JetBrains_Mono]">
                          {msg.model || "—"}
                        </span>
                        {msg.source && (
                          <>
                            <span className="text-[10px] text-text-muted">·</span>
                            <span className="text-[10px] text-text-muted uppercase">{msg.source}</span>
                          </>
                        )}
                        {!isUser && msg.tokens_per_second > 0 && (
                          <>
                            <span className="text-[10px] text-text-muted">·</span>
                            <span className="text-[10px] text-accent font-[JetBrains_Mono]">
                              {msg.tokens_per_second.toFixed(1)} tok/s
                            </span>
                          </>
                        )}
                        <span className="ml-auto text-[10px] text-text-muted">
                          {new Date(msg.timestamp * 1000).toLocaleTimeString("en-US", {
                            hour: "2-digit",
                            minute: "2-digit",
                            second: "2-digit",
                            hour12: false,
                          })}
                        </span>
                      </div>
                      <p className="text-sm text-text-primary/90 leading-relaxed line-clamp-3">
                        {msg.content}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
