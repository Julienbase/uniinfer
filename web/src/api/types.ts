export interface FitInfo {
  fits: boolean;
  model_size_gb: number;
  headroom_gb: number;
  warnings: string[];
}

export interface FallbackEvent {
  from: string;
  to: string;
  reason: string;
  success: boolean;
}

export interface FallbackInfo {
  fell_back: boolean;
  summary: string;
  events: FallbackEvent[];
}

export interface LastInference {
  method: string;
  elapsed_seconds: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tokens_per_second: number;
}

export interface DiagnosticsInfo {
  model_load_time_seconds: number;
  total_inferences: number;
  total_tokens_generated: number;
  total_inference_time_seconds: number;
  average_tokens_per_second: number;
  peak_tokens_per_second: number;
  last_inference: LastInference | null;
}

export interface SchedulerInfo {
  queue_depth: number;
  is_processing: boolean;
}

export interface StatusResponse {
  model: string;
  device: string;
  device_name: string;
  quantization: string;
  context_length: number;
  backend: string;
  model_path: string;
  loaded: boolean;
  device_memory_total_gb: number;
  device_memory_free_gb: number;
  uptime_seconds: number;
  fit: FitInfo | null;
  fallback: FallbackInfo | null;
  diagnostics: DiagnosticsInfo;
  scheduler: SchedulerInfo;
}

export interface DeviceResponse {
  name: string;
  device_string: string;
  device_type: string;
  total_memory_gb: number;
  free_memory_gb: number;
  is_active: boolean;
}

export interface CachedModel {
  model_id: string;
  quantization: string;
  file_size_bytes: number;
  file_size_gb: number;
  source: string;
  gguf_path: string;
  format: string; // "gguf" | "onnx" | "safetensors"
  is_loaded: boolean;
}

export interface ModelAlias {
  alias: string;
  display_name: string;
  repo_id: string;
  param_count_billions: number;
  default_quant: string;
  default_context_length: number;
  is_cached: boolean;
}

export interface ModelSizeResponse {
  model_id: string;
  quantization: string;
  filename: string | null;
  size_gb: number | null;
  fits: boolean | null;
  headroom_gb: number | null;
  error: string | null;
}

export interface DownloadProgress {
  status: "checking" | "downloading" | "complete" | "error";
  message: string;
  progress: number;
  downloaded_gb: number;
  total_gb: number;
  path?: string;
}

export interface ChatMessage {
  session_id: string;
  model: string;
  source: string;
  role: string;
  content: string;
  timestamp: number;
  tokens: number;
  tokens_per_second: number;
}

export interface ChatSessionSummary {
  session_id: string;
  model: string;
  source: string;
  created_at: number;
  message_count: number;
  last_message_preview: string;
}

export interface ChatSession {
  session_id: string;
  model: string;
  source: string;
  created_at: number;
  messages: ChatMessage[];
}

export interface ChatSummary {
  active_sessions: number;
  total_messages: number;
  last_message_preview: string;
}

export interface ModelLoadRequest {
  model_id: string;
  device?: string;
  quantization?: string;
  context_length?: number;
}

export interface ModelLoadResponse {
  success: boolean;
  model: string;
  device: string;
  quantization: string;
  backend: string;
  error: string | null;
}

export interface DashboardChatSendRequest {
  messages: { role: string; content: string }[];
  session_id?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  system_prompt?: string;
}

export interface DashboardGenerateRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
}

export interface DashboardGenerateResponse {
  text: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tokens_per_second: number;
  elapsed_seconds: number;
}

export interface BenchRunResult {
  run_number: number;
  tokens: number;
  elapsed_seconds: number;
  tokens_per_second: number;
}

export interface DashboardBenchRequest {
  prompt?: string;
  max_tokens?: number;
  runs?: number;
}

export interface DashboardBenchResponse {
  runs: BenchRunResult[];
  average_tokens_per_second: number;
  peak_tokens_per_second: number;
  total_tokens: number;
  model: string;
  device: string;
  quantization: string;
}

export interface FitAlternative {
  quantization: string;
  estimated_size_gb: number;
  fits: boolean;
}

export interface DashboardFitCheckRequest {
  model_id: string;
  quantization?: string;
  context_length?: number;
}

export interface DashboardFitCheckResponse {
  model_id: string;
  quantization: string;
  fits: boolean | null;
  model_size_gb: number | null;
  available_memory_gb: number | null;
  headroom_gb: number | null;
  overhead_gb: number | null;
  warnings: string[];
  alternatives: FitAlternative[];
  recommended_quantization: string | null;
  device_name: string;
  error: string | null;
}

export interface SSEEvent {
  type: string;
  uptime_seconds?: number;
  loaded?: boolean;
  device_memory_free_gb?: number;
  diagnostics?: {
    total_inferences: number;
    total_tokens_generated: number;
    average_tokens_per_second: number;
    peak_tokens_per_second: number;
  };
  scheduler?: {
    queue_depth: number;
    is_processing: boolean;
  };
  chat?: ChatSummary;
}
