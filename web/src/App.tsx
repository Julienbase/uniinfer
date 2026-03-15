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

const TAB_ICONS: Record<Tab, React.ReactNode> = {
  dashboard: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
      <rect x="1" y="1" width="6" height="6" rx="1" /><rect x="9" y="1" width="6" height="6" rx="1" />
      <rect x="1" y="9" width="6" height="6" rx="1" /><rect x="9" y="9" width="6" height="6" rx="1" />
    </svg>
  ),
  chat: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
      <path d="M2 3h12v8H5l-3 3V3z" strokeLinejoin="round" />
    </svg>
  ),
  generate: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
      <path d="M8 1v4M8 11v4M1 8h4M11 8h4M3 3l2.5 2.5M10.5 10.5L13 13M13 3l-2.5 2.5M5.5 10.5L3 13" strokeLinecap="round" />
    </svg>
  ),
  models: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
      <path d="M8 1L14 4.5V11.5L8 15L2 11.5V4.5L8 1z" strokeLinejoin="round" />
      <path d="M8 8V15M8 8L2 4.5M8 8L14 4.5" />
    </svg>
  ),
  devices: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
      <rect x="3" y="3" width="10" height="10" rx="1" />
      <path d="M6 1v2M10 1v2M6 13v2M10 13v2M1 6h2M1 10h2M13 6h2M13 10h2" strokeLinecap="round" />
    </svg>
  ),
  bench: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 4v4l3 2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  "fit-check": (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
      <circle cx="8" cy="8" r="6.5" />
      <circle cx="8" cy="8" r="2.5" />
      <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2" strokeLinecap="round" />
    </svg>
  ),
};

const tabs: { id: Tab; label: string; group?: string }[] = [
  { id: "dashboard", label: "Dashboard", group: "overview" },
  { id: "chat", label: "Chat", group: "inference" },
  { id: "generate", label: "Generate", group: "inference" },
  { id: "models", label: "Models", group: "manage" },
  { id: "devices", label: "Devices", group: "manage" },
  { id: "bench", label: "Bench", group: "tools" },
  { id: "fit-check", label: "Fit Check", group: "tools" },
];

function AppContent() {
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header */}
      <header className="border-b border-border bg-bg-secondary/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between h-14">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <svg viewBox="0 0 28 28" className="w-7 h-7" fill="none">
              <rect x="2" y="2" width="24" height="24" rx="6" fill="rgba(245,158,11,0.12)" stroke="rgba(245,158,11,0.4)" strokeWidth="1" />
              <path d="M8 10h4v8H8zM16 6h4v12h-4z" fill="var(--color-accent)" opacity="0.8" />
              <path d="M12 13h4v2h-4z" fill="var(--color-accent)" opacity="0.5" />
            </svg>
            <span className="font-display font-extrabold text-lg text-text-primary tracking-tight">
              UniInfer
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-accent-glow text-accent uppercase tracking-wider border border-accent/20">
              v1.5
            </span>
          </div>

          {/* Navigation */}
          <nav className="flex items-center">
            {tabs.map((tab, i) => {
              const prevGroup = i > 0 ? tabs[i - 1].group : null;
              const showSep = prevGroup && prevGroup !== tab.group;

              return (
                <div key={tab.id} className="flex items-center">
                  {showSep && (
                    <div className="w-px h-5 bg-border mx-1" />
                  )}
                  <button
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-all cursor-pointer relative ${
                      activeTab === tab.id
                        ? "text-accent font-medium"
                        : "text-text-muted hover:text-text-secondary"
                    }`}
                  >
                    {TAB_ICONS[tab.id]}
                    <span>{tab.label}</span>
                    {activeTab === tab.id && (
                      <span className="absolute bottom-0 left-3 right-3 h-0.5 bg-accent rounded-full" />
                    )}
                  </button>
                </div>
              );
            })}
          </nav>
        </div>
        {/* Signature accent line */}
        <div className="accent-line" />
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
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
