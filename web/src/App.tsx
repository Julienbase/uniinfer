import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DashboardPage } from "./pages/DashboardPage";
import { ModelsPage } from "./pages/ModelsPage";
import { DevicesPage } from "./pages/DevicesPage";
import { ChatPage } from "./pages/ChatPage";
import { GeneratePage } from "./pages/GeneratePage";
import { BenchPage } from "./pages/BenchPage";
import { FitCheckPage } from "./pages/FitCheckPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5000,
    },
  },
});

type Tab = "dashboard" | "models" | "devices" | "chat" | "generate" | "bench" | "fit-check";

const tabs: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "chat", label: "Chat" },
  { id: "generate", label: "Generate" },
  { id: "models", label: "Models" },
  { id: "devices", label: "Devices" },
  { id: "bench", label: "Bench" },
  { id: "fit-check", label: "Fit Check" },
];

function AppContent() {
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header */}
      <header className="border-b border-border bg-bg-secondary/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between h-14">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-accent/15 flex items-center justify-center">
              <span className="text-accent font-bold text-sm font-[JetBrains_Mono]">U</span>
            </div>
            <span className="font-semibold text-text-primary tracking-tight">
              UniInfer
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent-glow text-accent font-[JetBrains_Mono] uppercase tracking-wider">
              v1.5
            </span>
          </div>

          <nav className="flex gap-0.5">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-1.5 text-sm rounded-md transition-all cursor-pointer ${
                  activeTab === tab.id
                    ? "bg-accent/15 text-accent font-medium"
                    : "text-text-muted hover:text-text-secondary hover:bg-bg-card"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-6 py-6">
        {activeTab === "dashboard" && <DashboardPage />}
        {activeTab === "chat" && <ChatPage />}
        {activeTab === "generate" && <GeneratePage />}
        {activeTab === "models" && <ModelsPage />}
        {activeTab === "devices" && <DevicesPage />}
        {activeTab === "bench" && <BenchPage />}
        {activeTab === "fit-check" && <FitCheckPage />}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}
