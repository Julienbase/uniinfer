import { useState } from "react";
import { useCachedModels, useAliases, useDeleteModel } from "../api/hooks";
import { downloadModel, fetchModelSize, loadModel } from "../api/client";
import { formatGB } from "../utils/format";
import type { DownloadProgress, ModelSizeResponse } from "../api/types";
import { useQueryClient } from "@tanstack/react-query";

export function ModelsPage() {
  const { data: cached, isLoading: loadingCached } = useCachedModels();
  const { data: aliases, isLoading: loadingAliases } = useAliases();
  const deleteMutation = useDeleteModel();
  const queryClient = useQueryClient();

  const [downloadState, setDownloadState] = useState<DownloadProgress | null>(null);
  const [downloadingId, setDownloadingId] = useState<string>("");
  const [sizeCheck, setSizeCheck] = useState<ModelSizeResponse | null>(null);
  const [customModel, setCustomModel] = useState("");
  const [customQuant, setCustomQuant] = useState("q4_k_m");
  const [checkingSize, setCheckingSize] = useState(false);
  const [loadingModelId, setLoadingModelId] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);

  async function handleDownload(modelId: string, quant: string) {
    setDownloadingId(`${modelId}::${quant}`);
    setDownloadState({ status: "checking", message: "Starting...", progress: 0, downloaded_gb: 0, total_gb: 0 });
    try {
      await downloadModel(modelId, quant, (p) => setDownloadState(p));
    } catch {
      setDownloadState({ status: "error", message: "Download failed", progress: 0, downloaded_gb: 0, total_gb: 0 });
    }
    // Refresh cached models list
    queryClient.invalidateQueries({ queryKey: ["cached-models"] });
    queryClient.invalidateQueries({ queryKey: ["aliases"] });
    setTimeout(() => {
      setDownloadingId("");
      setDownloadState(null);
    }, 3000);
  }

  async function handleCheckSize() {
    if (!customModel.trim()) return;
    setCheckingSize(true);
    try {
      const result = await fetchModelSize(customModel.trim(), customQuant);
      setSizeCheck(result);
    } catch {
      setSizeCheck(null);
    }
    setCheckingSize(false);
  }

  async function handleLoadModel(modelId: string, quantization: string) {
    const key = `${modelId}::${quantization}`;
    setLoadingModelId(key);
    setLoadError(null);

    try {
      const result = await loadModel({ model_id: modelId, quantization });
      if (!result.success) {
        setLoadError(result.error || "Failed to load model");
      } else {
        // Refresh all data
        queryClient.invalidateQueries({ queryKey: ["status"] });
        queryClient.invalidateQueries({ queryKey: ["cached-models"] });
      }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Load failed");
    } finally {
      setLoadingModelId("");
    }
  }

  return (
    <div className="space-y-6">
      {/* Load error banner */}
      {loadError && (
        <div className="bg-danger-dim border border-danger/30 rounded-xl p-4 animate-fade-in flex items-center justify-between">
          <p className="text-sm text-danger">{loadError}</p>
          <button
            onClick={() => setLoadError(null)}
            className="text-danger/60 hover:text-danger text-sm cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Loading model indicator */}
      {loadingModelId && (
        <div className="bg-accent/8 border border-accent/20 rounded-xl p-4 animate-fade-in">
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-accent font-medium">
              Loading model... This may take a moment.
            </span>
          </div>
        </div>
      )}

      {/* Cached Models */}
      <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary mb-4">
          Cached Models
        </h2>

        {loadingCached ? (
          <div className="text-text-muted text-sm py-4 text-center">Loading...</div>
        ) : !cached?.length ? (
          <div className="text-text-muted text-sm py-4 text-center">
            No models cached. Download one from the aliases below.
          </div>
        ) : (
          <div className="space-y-2">
            {cached.map((m) => {
              const key = `${m.model_id}::${m.quantization}`;
              const isLoading = loadingModelId === key;

              return (
                <div
                  key={key}
                  className={`flex items-center justify-between p-3 rounded-lg bg-bg-secondary border transition-colors ${
                    m.is_loaded
                      ? "border-success/30 bg-success/5"
                      : "border-border hover:border-border-bright"
                  }`}
                >
                  <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-[JetBrains_Mono] text-text-primary truncate">
                        {m.model_id}
                      </span>
                      {m.is_loaded && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-success-dim text-success font-semibold uppercase">
                          Active
                        </span>
                      )}
                    </div>
                    <div className="flex gap-3 text-xs text-text-muted items-center">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                        m.format === "onnx" ? "bg-blue-500/15 text-blue-400" :
                        m.format === "safetensors" ? "bg-purple-500/15 text-purple-400" :
                        "bg-text-muted/10 text-text-muted"
                      }`}>{m.format}</span>
                      <span>{m.quantization.toUpperCase()}</span>
                      <span>{formatGB(m.file_size_gb)}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {!m.is_loaded && (
                      <>
                        <button
                          className="text-xs px-3 py-1.5 rounded-md bg-accent-glow text-accent hover:bg-accent/20 transition-colors cursor-pointer disabled:opacity-50"
                          onClick={() => handleLoadModel(m.model_id, m.quantization)}
                          disabled={!!loadingModelId}
                        >
                          {isLoading ? (
                            <span className="flex items-center gap-1.5">
                              <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                              Loading
                            </span>
                          ) : (
                            "Load"
                          )}
                        </button>
                        <button
                          className="text-xs px-3 py-1.5 rounded-md bg-danger-dim text-danger hover:bg-danger/20 transition-colors cursor-pointer"
                          onClick={() => deleteMutation.mutate({ modelId: m.model_id, quantization: m.quantization, format: m.format })}
                          disabled={deleteMutation.isPending || !!loadingModelId}
                        >
                          {deleteMutation.isPending ? "..." : "Delete"}
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Download Progress */}
      {downloadState && (
        <div className="bg-bg-card border border-accent/30 rounded-xl p-5 animate-fade-in">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-primary font-semibold">
              {downloadState.status === "complete" ? "Download Complete" :
               downloadState.status === "error" ? "Download Failed" :
               "Downloading..."}
            </span>
            <span className="text-xs text-text-muted font-[JetBrains_Mono]">
              {downloadState.status === "downloading"
                ? `${formatGB(downloadState.downloaded_gb)} / ${formatGB(downloadState.total_gb)}`
                : downloadState.message}
            </span>
          </div>
          <div className="w-full h-2 bg-bg-secondary rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                downloadState.status === "error" ? "bg-danger" :
                downloadState.status === "complete" ? "bg-success" :
                "bg-accent"
              }`}
              style={{ width: `${downloadState.progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Model Aliases */}
      <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary mb-4">
          Model Aliases
        </h2>

        {loadingAliases ? (
          <div className="text-text-muted text-sm py-4 text-center">Loading...</div>
        ) : (
          <div className="space-y-2">
            {aliases?.map((a) => (
              <div
                key={a.alias}
                className="flex items-center justify-between p-3 rounded-lg bg-bg-secondary border border-border hover:border-border-bright transition-colors"
              >
                <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-text-primary">
                      {a.display_name}
                    </span>
                    <span className="text-xs font-[JetBrains_Mono] text-text-muted">
                      {a.alias}
                    </span>
                  </div>
                  <div className="flex gap-3 text-xs text-text-muted">
                    <span>{a.param_count_billions}B params</span>
                    <span>{a.default_quant.toUpperCase()}</span>
                    <span>{a.default_context_length.toLocaleString()} ctx</span>
                  </div>
                </div>

                {a.is_cached ? (
                  <span className="text-[10px] px-2 py-1 rounded bg-success-dim text-success font-semibold uppercase">
                    Cached
                  </span>
                ) : (
                  <button
                    className="text-xs px-3 py-1.5 rounded-md bg-accent-glow text-accent hover:bg-accent/20 transition-colors cursor-pointer disabled:opacity-50"
                    onClick={() => handleDownload(a.repo_id, a.default_quant)}
                    disabled={downloadingId === `${a.repo_id}::${a.default_quant}`}
                  >
                    Download
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Custom Model Download */}
      <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary mb-4">
          Custom Model
        </h2>

        <div className="flex gap-3">
          <input
            type="text"
            placeholder="HuggingFace model ID (GGUF, ONNX, or SafeTensors)"
            value={customModel}
            onChange={(e) => setCustomModel(e.target.value)}
            className="flex-1 px-3 py-2 text-sm bg-bg-input border border-border rounded-lg text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent transition-colors"
          />
          <select
            value={customQuant}
            onChange={(e) => setCustomQuant(e.target.value)}
            className="px-3 py-2 text-sm bg-bg-input border border-border rounded-lg text-text-primary focus:outline-none focus:border-accent cursor-pointer"
          >
            {["f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q3_k_m", "q2_k"].map((q) => (
              <option key={q} value={q}>{q.toUpperCase()}</option>
            ))}
          </select>
          <button
            onClick={handleCheckSize}
            disabled={checkingSize || !customModel.trim()}
            className="px-4 py-2 text-sm bg-bg-secondary border border-border rounded-lg text-text-secondary hover:text-text-primary hover:border-border-bright transition-colors cursor-pointer disabled:opacity-50"
          >
            {checkingSize ? "..." : "Check"}
          </button>
          <button
            onClick={() => handleDownload(customModel.trim(), customQuant)}
            disabled={!customModel.trim() || !!downloadingId}
            className="px-4 py-2 text-sm bg-accent-dim text-white rounded-lg hover:bg-accent transition-colors cursor-pointer disabled:opacity-50"
          >
            Download
          </button>
        </div>

        {sizeCheck && (
          <div className="mt-3 p-3 rounded-lg bg-bg-secondary border border-border text-sm">
            {sizeCheck.error ? (
              <span className="text-danger">{sizeCheck.error}</span>
            ) : (
              <div className="flex gap-4 text-text-secondary">
                <span className="font-[JetBrains_Mono] text-text-primary">
                  {sizeCheck.filename}
                </span>
                <span>{formatGB(sizeCheck.size_gb!)}</span>
                {sizeCheck.fits !== null && (
                  <span className={sizeCheck.fits ? "text-success" : "text-danger"}>
                    {sizeCheck.fits ? "Fits" : "Won't fit"}
                    {sizeCheck.headroom_gb !== null && ` (${formatGB(sizeCheck.headroom_gb)} headroom)`}
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
