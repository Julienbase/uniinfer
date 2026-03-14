"""Pydantic schemas for the dashboard API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------


class FitInfo(BaseModel):
    fits: bool
    model_size_gb: float
    headroom_gb: float
    warnings: list[str] = Field(default_factory=list)


class FallbackEvent(BaseModel):
    from_device: str = Field(alias="from")
    to_device: str = Field(alias="to")
    reason: str
    success: bool

    model_config = {"populate_by_name": True}


class FallbackInfo(BaseModel):
    fell_back: bool
    summary: str
    events: list[FallbackEvent] = Field(default_factory=list)


class LastInference(BaseModel):
    method: str = ""
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0


class DiagnosticsInfo(BaseModel):
    model_load_time_seconds: float = 0.0
    total_inferences: int = 0
    total_tokens_generated: int = 0
    total_inference_time_seconds: float = 0.0
    average_tokens_per_second: float = 0.0
    peak_tokens_per_second: float = 0.0
    last_inference: Optional[LastInference] = None


class SchedulerInfo(BaseModel):
    queue_depth: int = 0
    is_processing: bool = False


class StatusResponse(BaseModel):
    model: str = ""
    device: str = ""
    device_name: str = ""
    quantization: str = ""
    context_length: int = 0
    backend: str = ""
    model_path: str = ""
    loaded: bool = False
    device_memory_total_gb: float = 0.0
    device_memory_free_gb: float = 0.0
    uptime_seconds: float = 0.0
    fit: Optional[FitInfo] = None
    fallback: Optional[FallbackInfo] = None
    diagnostics: DiagnosticsInfo = Field(default_factory=DiagnosticsInfo)
    scheduler: SchedulerInfo = Field(default_factory=SchedulerInfo)


# ---------------------------------------------------------------------------
# Device listing
# ---------------------------------------------------------------------------


class DeviceResponse(BaseModel):
    name: str
    device_string: str
    device_type: str
    total_memory_gb: float
    free_memory_gb: float
    is_active: bool = False


class DeviceListResponse(BaseModel):
    devices: list[DeviceResponse]


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


class CachedModelResponse(BaseModel):
    model_id: str
    quantization: str
    file_size_bytes: int
    file_size_gb: float
    source: str
    gguf_path: str
    is_loaded: bool = False


class CachedModelsListResponse(BaseModel):
    models: list[CachedModelResponse]


class AliasResponse(BaseModel):
    alias: str
    display_name: str
    repo_id: str
    param_count_billions: float
    default_quant: str
    default_context_length: int
    is_cached: bool = False


class AliasListResponse(BaseModel):
    aliases: list[AliasResponse]


class ModelSizeResponse(BaseModel):
    model_id: str
    quantization: str
    filename: Optional[str] = None
    size_gb: Optional[float] = None
    fits: Optional[bool] = None
    headroom_gb: Optional[float] = None
    error: Optional[str] = None


class ModelDeleteRequest(BaseModel):
    model_id: str
    quantization: str


class ModelDeleteResponse(BaseModel):
    deleted: bool
    freed_bytes: int = 0


class ModelDownloadRequest(BaseModel):
    model_id: str
    quantization: str = "q4_k_m"


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------


class ChatMessageResponse(BaseModel):
    session_id: str = ""
    model: str = ""
    source: str = ""
    role: str
    content: str
    timestamp: float
    tokens: int = 0
    tokens_per_second: float = 0.0


class ChatSessionSummary(BaseModel):
    session_id: str
    model: str
    source: str
    created_at: float
    message_count: int
    last_message_preview: str = ""


class ChatSessionResponse(BaseModel):
    session_id: str
    model: str
    source: str
    created_at: float
    messages: list[ChatMessageResponse]


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionSummary]


class RecentMessagesResponse(BaseModel):
    messages: list[ChatMessageResponse]


class ChatSummary(BaseModel):
    active_sessions: int = 0
    total_messages: int = 0
    last_message_preview: str = ""


# ---------------------------------------------------------------------------
# Dashboard chat send (interactive chat from browser)
# ---------------------------------------------------------------------------


class ModelLoadRequest(BaseModel):
    model_id: str
    device: str = "auto"
    quantization: str = "auto"
    context_length: int = 4096


class ModelLoadResponse(BaseModel):
    success: bool
    model: str = ""
    device: str = ""
    quantization: str = ""
    backend: str = ""
    error: Optional[str] = None


class DashboardChatMessage(BaseModel):
    role: str
    content: str


class DashboardChatSendRequest(BaseModel):
    messages: list[DashboardChatMessage]
    session_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    system_prompt: Optional[str] = None


class DashboardGenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[list[str]] = None


class DashboardGenerateResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    elapsed_seconds: float = 0.0


class DashboardBenchRequest(BaseModel):
    prompt: str = "Explain the theory of general relativity in simple terms."
    max_tokens: int = 128
    runs: int = 3


class BenchRunResult(BaseModel):
    run_number: int
    tokens: int
    elapsed_seconds: float
    tokens_per_second: float


class DashboardBenchResponse(BaseModel):
    runs: list[BenchRunResult]
    average_tokens_per_second: float
    peak_tokens_per_second: float
    total_tokens: int
    model: str = ""
    device: str = ""
    quantization: str = ""


class FitAlternative(BaseModel):
    quantization: str
    estimated_size_gb: float
    fits: bool


class DashboardFitCheckRequest(BaseModel):
    model_id: str
    quantization: str = "q4_k_m"
    context_length: int = 4096


class DashboardFitCheckResponse(BaseModel):
    model_id: str
    quantization: str
    fits: Optional[bool] = None
    model_size_gb: Optional[float] = None
    available_memory_gb: Optional[float] = None
    headroom_gb: Optional[float] = None
    overhead_gb: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    alternatives: list[FitAlternative] = Field(default_factory=list)
    recommended_quantization: Optional[str] = None
    device_name: str = ""
    error: Optional[str] = None
