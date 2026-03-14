const BASE_URL = "/api/dashboard";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export async function fetchStatus() {
  return fetchJSON<import("./types").StatusResponse>("/status");
}

export async function fetchDevices() {
  return fetchJSON<{ devices: import("./types").DeviceResponse[] }>("/devices");
}

export async function fetchCachedModels() {
  return fetchJSON<{ models: import("./types").CachedModel[] }>("/models/cached");
}

export async function fetchAliases() {
  return fetchJSON<{ aliases: import("./types").ModelAlias[] }>("/models/aliases");
}

export async function fetchModelSize(modelId: string, quantization: string = "q4_k_m") {
  return fetchJSON<import("./types").ModelSizeResponse>(
    `/models/size?model_id=${encodeURIComponent(modelId)}&quantization=${encodeURIComponent(quantization)}`
  );
}

export async function deleteModel(modelId: string, quantization: string) {
  return fetchJSON<{ deleted: boolean; freed_bytes: number }>("/models/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, quantization }),
  });
}

export async function loadModel(request: import("./types").ModelLoadRequest) {
  return fetchJSON<import("./types").ModelLoadResponse>("/models/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function fetchChatSessions() {
  return fetchJSON<{ sessions: import("./types").ChatSessionSummary[] }>("/chat/sessions");
}

export async function fetchChatSession(sessionId: string) {
  return fetchJSON<import("./types").ChatSession>(`/chat/sessions/${encodeURIComponent(sessionId)}`);
}

export async function fetchRecentMessages(limit = 50) {
  return fetchJSON<{ messages: import("./types").ChatMessage[] }>(`/chat/recent?limit=${limit}`);
}

export async function sendChatMessage(
  request: import("./types").DashboardChatSendRequest,
  onToken: (token: string) => void,
  onDone: (sessionId: string) => void,
  signal?: AbortSignal,
) {
  const res = await fetch(`${BASE_URL}/chat/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!res.ok || !res.body) {
    const body = await res.text();
    throw new Error(`Chat request failed: ${res.status} — ${body}`);
  }

  const sessionId = res.headers.get("X-Session-Id") || "";
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const payload = line.slice(6);
        if (payload === "[DONE]") {
          onDone(sessionId);
          return;
        }
        try {
          const data = JSON.parse(payload);
          const content = data?.choices?.[0]?.delta?.content;
          if (content) onToken(content);
        } catch {
          // skip malformed
        }
      }
    }
  }
  onDone(sessionId);
}

export async function dashboardGenerate(
  request: import("./types").DashboardGenerateRequest,
) {
  return fetchJSON<import("./types").DashboardGenerateResponse>("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function runBenchmark(
  request: import("./types").DashboardBenchRequest,
) {
  return fetchJSON<import("./types").DashboardBenchResponse>("/bench", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function runFitCheck(
  request: import("./types").DashboardFitCheckRequest,
) {
  return fetchJSON<import("./types").DashboardFitCheckResponse>("/fit-check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function downloadModel(
  modelId: string,
  quantization: string,
  onProgress: (p: import("./types").DownloadProgress) => void,
) {
  const res = await fetch(`${BASE_URL}/models/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, quantization }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Download request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          onProgress(data);
        } catch {
          // skip malformed lines
        }
      }
    }
  }
}
