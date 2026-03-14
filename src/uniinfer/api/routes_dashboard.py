"""Dashboard API routes for model management, monitoring, and diagnostics."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from uniinfer.api.dashboard_schemas import (
    AliasListResponse,
    AliasResponse,
    BenchRunResult,
    CachedModelResponse,
    CachedModelsListResponse,
    ChatMessageResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionSummary,
    DashboardBenchRequest,
    DashboardBenchResponse,
    DashboardChatSendRequest,
    DashboardFitCheckRequest,
    DashboardFitCheckResponse,
    DashboardGenerateRequest,
    DashboardGenerateResponse,
    DeviceListResponse,
    DeviceResponse,
    DiagnosticsInfo,
    FallbackEvent,
    FallbackInfo,
    FitAlternative,
    FitInfo,
    LastInference,
    ModelDeleteRequest,
    ModelDeleteResponse,
    ModelDownloadRequest,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelSizeResponse,
    RecentMessagesResponse,
    SchedulerInfo,
    StatusResponse,
)
from uniinfer.api.download_manager import DownloadManager

if TYPE_CHECKING:
    from uniinfer.api.server import UniInferServer

logger = logging.getLogger(__name__)

# Shared download manager instance
_download_manager = DownloadManager()


def create_dashboard_router(server: UniInferServer, start_time: float) -> APIRouter:
    """Create the dashboard router with all management/monitoring endpoints.

    Args:
        server: The UniInferServer instance.
        start_time: Server start timestamp (time.time()).

    Returns:
        FastAPI APIRouter with /api/dashboard/* endpoints.
    """
    router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

    # ------------------------------------------------------------------
    # GET /api/dashboard/status — full system status
    # ------------------------------------------------------------------
    @router.get("/status")
    async def get_status() -> StatusResponse:
        if server.engine is None:
            return StatusResponse(uptime_seconds=time.time() - start_time)

        info = server.engine.info()
        diag = info.get("diagnostics", {})
        last = diag.get("last_inference")

        fit_data = info.get("fit")
        fit_info = FitInfo(**fit_data) if fit_data else None

        fallback_data = info.get("fallback")
        fallback_info = None
        if fallback_data:
            events = [
                FallbackEvent(**{
                    "from": e["from"],
                    "to": e["to"],
                    "reason": e["reason"],
                    "success": e["success"],
                })
                for e in fallback_data.get("events", [])
            ]
            fallback_info = FallbackInfo(
                fell_back=fallback_data["fell_back"],
                summary=fallback_data["summary"],
                events=events,
            )

        return StatusResponse(
            model=info.get("model", ""),
            device=info.get("device", ""),
            device_name=info.get("device_name", ""),
            quantization=info.get("quantization", ""),
            context_length=info.get("context_length", 0),
            backend=info.get("backend", ""),
            model_path=info.get("model_path", ""),
            loaded=info.get("loaded", False),
            device_memory_total_gb=info.get("device_memory_total_gb", 0.0),
            device_memory_free_gb=info.get("device_memory_free_gb", 0.0),
            uptime_seconds=time.time() - start_time,
            fit=fit_info,
            fallback=fallback_info,
            diagnostics=DiagnosticsInfo(
                model_load_time_seconds=diag.get("model_load_time_seconds", 0.0),
                total_inferences=diag.get("total_inferences", 0),
                total_tokens_generated=diag.get("total_tokens_generated", 0),
                total_inference_time_seconds=diag.get("total_inference_time_seconds", 0.0),
                average_tokens_per_second=diag.get("average_tokens_per_second", 0.0),
                peak_tokens_per_second=diag.get("peak_tokens_per_second", 0.0),
                last_inference=LastInference(**last) if last else None,
            ),
            scheduler=SchedulerInfo(
                queue_depth=server.scheduler.queue_depth if server.scheduler else 0,
                is_processing=server.scheduler.is_processing if server.scheduler else False,
            ),
        )

    # ------------------------------------------------------------------
    # GET /api/dashboard/events — SSE stream for real-time updates
    # ------------------------------------------------------------------
    @router.get("/events")
    async def dashboard_events(request: Request) -> StreamingResponse:
        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break

                data: dict = {"type": "heartbeat"}
                if server.engine is not None:
                    info = server.engine.info()
                    diag = info.get("diagnostics", {})
                    data = {
                        "type": "status",
                        "uptime_seconds": round(time.time() - start_time, 1),
                        "loaded": info.get("loaded", False),
                        "device_memory_free_gb": info.get("device_memory_free_gb", 0.0),
                        "diagnostics": {
                            "total_inferences": diag.get("total_inferences", 0),
                            "total_tokens_generated": diag.get("total_tokens_generated", 0),
                            "average_tokens_per_second": diag.get("average_tokens_per_second", 0.0),
                            "peak_tokens_per_second": diag.get("peak_tokens_per_second", 0.0),
                        },
                        "scheduler": {
                            "queue_depth": server.scheduler.queue_depth if server.scheduler else 0,
                            "is_processing": server.scheduler.is_processing if server.scheduler else False,
                        },
                        "chat": server.chat_store.summary(),
                    }

                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # GET /api/dashboard/devices — available hardware
    # ------------------------------------------------------------------
    @router.get("/devices")
    async def list_devices() -> DeviceListResponse:
        from uniinfer.hal.discovery import devices as discover_devices

        device_list: list[DeviceResponse] = []
        active_device = ""

        if server.engine is not None:
            info = server.engine.info()
            active_device = info.get("device", "")
            hw_devices = server.engine.available_devices
        else:
            # No engine loaded — discover hardware directly
            try:
                hw_devices = discover_devices()
            except Exception:
                hw_devices = []

        for dev in hw_devices:
            device_list.append(DeviceResponse(
                name=dev.name,
                device_string=dev.device_string,
                device_type=dev.device_type.value if hasattr(dev.device_type, "value") else str(dev.device_type),
                total_memory_gb=round(dev.total_memory_gb, 2),
                free_memory_gb=round(dev.free_memory_gb, 2),
                is_active=(dev.device_string == active_device),
            ))

        return DeviceListResponse(devices=device_list)

    # ------------------------------------------------------------------
    # GET /api/dashboard/models/cached — cached models list
    # ------------------------------------------------------------------
    @router.get("/models/cached")
    async def list_cached_models() -> CachedModelsListResponse:
        from uniinfer.models.registry import list_cached

        cached = list_cached()
        loaded_model = ""
        if server.engine is not None:
            loaded_model = server.engine.info().get("model", "")

        models = [
            CachedModelResponse(
                model_id=m.model_id,
                quantization=m.quantization,
                file_size_bytes=m.file_size,
                file_size_gb=round(m.file_size / (1024**3), 2),
                source=m.source,
                gguf_path=m.gguf_path,
                is_loaded=(m.model_id == loaded_model),
            )
            for m in cached
        ]

        return CachedModelsListResponse(models=models)

    # ------------------------------------------------------------------
    # GET /api/dashboard/models/aliases — available model aliases
    # ------------------------------------------------------------------
    @router.get("/models/aliases")
    async def list_model_aliases() -> AliasListResponse:
        from uniinfer.models.aliases import list_aliases
        from uniinfer.models.registry import is_cached

        aliases = [
            AliasResponse(
                alias=name,
                display_name=info.display_name,
                repo_id=info.repo_id,
                param_count_billions=info.param_count_billions,
                default_quant=info.default_quant,
                default_context_length=info.default_context_length,
                is_cached=is_cached(info.repo_id, info.default_quant),
            )
            for name, info in list_aliases()
        ]

        return AliasListResponse(aliases=aliases)

    # ------------------------------------------------------------------
    # GET /api/dashboard/models/size — pre-download size/fit check
    # ------------------------------------------------------------------
    @router.get("/models/size")
    async def check_model_size(model_id: str, quantization: str = "q4_k_m") -> ModelSizeResponse:
        from uniinfer.models.registry import query_model_size_from_hf

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, query_model_size_from_hf, model_id, quantization,
            )
        except Exception as exc:
            return ModelSizeResponse(
                model_id=model_id,
                quantization=quantization,
                error=str(exc),
            )

        if result is None:
            return ModelSizeResponse(
                model_id=model_id,
                quantization=quantization,
                error="No GGUF files found for this model on HuggingFace",
            )

        filename, size_gb = result

        # Run fit check if engine has device info
        fits = None
        headroom = None
        if server.engine is not None:
            try:
                from uniinfer.models.fitting import check_model_fit

                info = server.engine.info()
                devices = server.engine.available_devices
                if devices:
                    active = info.get("device", "")
                    device = next(
                        (d for d in devices if d.device_string == active),
                        devices[0],
                    )
                    report = check_model_fit(
                        device=device,
                        model_size_gb=size_gb,
                        quantization=quantization,
                    )
                    fits = report.fits
                    headroom = round(report.headroom_gb, 2)
            except Exception:
                pass

        return ModelSizeResponse(
            model_id=model_id,
            quantization=quantization,
            filename=filename,
            size_gb=round(size_gb, 2),
            fits=fits,
            headroom_gb=headroom,
        )

    # ------------------------------------------------------------------
    # DELETE /api/dashboard/models/cached — delete a cached model
    # ------------------------------------------------------------------
    @router.post("/models/delete")
    async def delete_model(req: ModelDeleteRequest) -> ModelDeleteResponse:
        from uniinfer.models.registry import delete_cached

        # Prevent deleting the currently loaded model
        if server.engine is not None:
            loaded = server.engine.info().get("model", "")
            if req.model_id == loaded:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete the currently loaded model. Unload it first.",
                )

        try:
            freed = delete_cached(req.model_id, req.quantization)
            return ModelDeleteResponse(deleted=True, freed_bytes=freed)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # POST /api/dashboard/models/load — hot-swap to a different model
    # ------------------------------------------------------------------
    @router.post("/models/load")
    async def load_model(req: ModelLoadRequest) -> ModelLoadResponse:
        """Load/switch to a different model (hot-swap)."""
        try:
            info = await server.swap_model(
                model=req.model_id,
                device=req.device,
                quantization=req.quantization,
                context_length=req.context_length,
            )
            return ModelLoadResponse(
                success=True,
                model=info.get("model", ""),
                device=info.get("device", ""),
                quantization=info.get("quantization", ""),
                backend=info.get("backend", ""),
            )
        except RuntimeError as exc:
            return ModelLoadResponse(success=False, error=str(exc))
        except Exception as exc:
            logger.error("Model load failed: %s", exc)
            return ModelLoadResponse(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # POST /api/dashboard/models/download — download with SSE progress
    # ------------------------------------------------------------------
    @router.post("/models/download")
    async def download_model(req: ModelDownloadRequest) -> StreamingResponse:
        return StreamingResponse(
            _download_manager.download_with_progress(
                model_id=req.model_id,
                quantization=req.quantization,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Chat history endpoints
    # ------------------------------------------------------------------
    @router.get("/chat/sessions")
    async def list_chat_sessions() -> ChatSessionListResponse:
        sessions = server.chat_store.list_sessions()
        return ChatSessionListResponse(
            sessions=[ChatSessionSummary(**s) for s in sessions]
        )

    @router.get("/chat/sessions/{session_id}")
    async def get_chat_session(session_id: str) -> ChatSessionResponse:
        session = server.chat_store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return ChatSessionResponse(
            session_id=session.session_id,
            model=session.model,
            source=session.source,
            created_at=session.created_at,
            messages=[
                ChatMessageResponse(
                    session_id=session.session_id,
                    model=session.model,
                    source=session.source,
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                    tokens=m.tokens,
                    tokens_per_second=m.tokens_per_second,
                )
                for m in session.messages
            ],
        )

    @router.get("/chat/recent")
    async def get_recent_messages(limit: int = 50) -> RecentMessagesResponse:
        messages = server.chat_store.get_recent_messages(limit=min(limit, 200))
        return RecentMessagesResponse(
            messages=[ChatMessageResponse(**m) for m in messages]
        )

    # ------------------------------------------------------------------
    # POST /api/dashboard/chat/send — interactive chat from dashboard
    # ------------------------------------------------------------------
    @router.post("/chat/send")
    async def dashboard_chat_send(req: DashboardChatSendRequest) -> StreamingResponse:
        """Send a chat message from the dashboard and stream the response.

        This endpoint bypasses API key auth (dashboard routes are exempt)
        and streams an SSE response with the same format as /v1/chat/completions.
        Messages are automatically stored in the chat history.
        """
        from uniinfer.api.chat_store import ChatMessage as StoreChatMessage
        from uniinfer.api.schemas import _generate_id
        from uniinfer.api.streaming import chat_stream_to_sse
        from uniinfer.engine.request import InferenceRequest

        if server.engine is None or server.scheduler is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        model_name = server.engine.info().get("model", "")
        request_id = _generate_id("chatcmpl")

        # Build messages list, prepending system prompt if provided
        messages: list[dict[str, str]] = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        for m in req.messages:
            messages.append({"role": m.role, "content": m.content})

        inference_req = InferenceRequest(
            request_id=request_id,
            messages=messages,
            is_chat=True,
            stream=True,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

        try:
            await server.scheduler.add_request(inference_req)
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Server is overloaded. Try again later.",
            )

        # Get or create session for chat store
        session_id = server.chat_store.get_or_create_session(
            req.session_id, model_name, "dashboard"
        )

        chunks = server.scheduler.stream_result(request_id)

        async def tracked_stream():
            full_text = ""
            token_count = 0
            stream_start = time.time()

            async for line in chat_stream_to_sse(request_id, model_name, chunks):
                yield line
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            token_count += 1
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

            # Store messages in chat history
            elapsed = time.time() - stream_start
            for m in messages:
                if m["role"] in ("user", "system"):
                    server.chat_store.add_message(
                        session_id,
                        StoreChatMessage(role=m["role"], content=m["content"]),
                    )
            tok_s = token_count / elapsed if elapsed > 0 else 0.0
            server.chat_store.add_message(
                session_id,
                StoreChatMessage(
                    role="assistant",
                    content=full_text,
                    tokens=token_count,
                    tokens_per_second=round(tok_s, 1),
                ),
            )

        return StreamingResponse(
            tracked_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Session-Id": session_id,
            },
        )

    # ------------------------------------------------------------------
    # POST /api/dashboard/generate — one-shot text generation
    # ------------------------------------------------------------------
    @router.post("/generate")
    async def dashboard_generate(req: DashboardGenerateRequest) -> DashboardGenerateResponse:
        """Generate text from a prompt (one-shot, non-streaming)."""
        from uniinfer.api.schemas import _generate_id
        from uniinfer.engine.request import InferenceRequest

        if server.engine is None or server.scheduler is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        request_id = _generate_id("cmpl")
        inference_req = InferenceRequest(
            request_id=request_id,
            prompt=req.prompt,
            is_chat=False,
            stream=False,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )

        try:
            await server.scheduler.add_request(inference_req)
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Server is overloaded. Try again later.",
            )

        gen_start = time.time()
        try:
            result = await server.scheduler.get_result(request_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed = time.time() - gen_start
        tok_s = result.completion_tokens / elapsed if elapsed > 0 else 0.0

        return DashboardGenerateResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            tokens_per_second=round(tok_s, 1),
            elapsed_seconds=round(elapsed, 3),
        )

    # ------------------------------------------------------------------
    # POST /api/dashboard/bench — run benchmark (N inference runs)
    # ------------------------------------------------------------------
    @router.post("/bench")
    async def dashboard_bench(req: DashboardBenchRequest) -> DashboardBenchResponse:
        """Run a benchmark: N identical inference runs, measure tok/s."""
        from uniinfer.api.schemas import _generate_id
        from uniinfer.engine.request import InferenceRequest

        if server.engine is None or server.scheduler is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        info = server.engine.info()
        runs: list[BenchRunResult] = []

        for i in range(min(req.runs, 10)):  # Cap at 10 runs
            request_id = _generate_id("bench")
            inference_req = InferenceRequest(
                request_id=request_id,
                prompt=req.prompt,
                is_chat=False,
                stream=False,
                max_tokens=req.max_tokens,
                temperature=0.0,  # Greedy for consistent benchmark
            )

            try:
                await server.scheduler.add_request(inference_req)
            except RuntimeError:
                raise HTTPException(
                    status_code=503,
                    detail="Server is overloaded. Try again later.",
                )

            run_start = time.time()
            try:
                result = await server.scheduler.get_result(request_id)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

            elapsed = time.time() - run_start
            tps = result.completion_tokens / elapsed if elapsed > 0 else 0.0

            runs.append(BenchRunResult(
                run_number=i + 1,
                tokens=result.completion_tokens,
                elapsed_seconds=round(elapsed, 3),
                tokens_per_second=round(tps, 1),
            ))

        tps_values = [r.tokens_per_second for r in runs]
        total_tokens = sum(r.tokens for r in runs)

        return DashboardBenchResponse(
            runs=runs,
            average_tokens_per_second=round(sum(tps_values) / len(tps_values), 1) if tps_values else 0.0,
            peak_tokens_per_second=round(max(tps_values), 1) if tps_values else 0.0,
            total_tokens=total_tokens,
            model=info.get("model", ""),
            device=info.get("device", ""),
            quantization=info.get("quantization", ""),
        )

    # ------------------------------------------------------------------
    # POST /api/dashboard/fit-check — comprehensive fit check
    # ------------------------------------------------------------------
    @router.post("/fit-check")
    async def dashboard_fit_check(req: DashboardFitCheckRequest) -> DashboardFitCheckResponse:
        """Run a comprehensive fit check for a model on current hardware."""
        from uniinfer.hal.discovery import devices as discover_devices
        from uniinfer.hal.discovery import select_best_device
        from uniinfer.models.registry import query_model_size_from_hf

        loop = asyncio.get_running_loop()

        # Query model size from HuggingFace
        try:
            result = await loop.run_in_executor(
                None, query_model_size_from_hf, req.model_id, req.quantization,
            )
        except Exception as exc:
            return DashboardFitCheckResponse(
                model_id=req.model_id,
                quantization=req.quantization,
                error=str(exc),
            )

        if result is None:
            return DashboardFitCheckResponse(
                model_id=req.model_id,
                quantization=req.quantization,
                error="No GGUF files found for this model on HuggingFace",
            )

        _filename, size_gb = result

        # Get device info from engine if loaded, otherwise discover directly
        if server.engine is not None:
            info = server.engine.info()
            hw_devices = server.engine.available_devices
            active_device_str = info.get("device", "")
        else:
            try:
                hw_devices = discover_devices()
            except Exception:
                hw_devices = []
            active_device_str = ""

        if not hw_devices:
            return DashboardFitCheckResponse(
                model_id=req.model_id,
                quantization=req.quantization,
                model_size_gb=round(size_gb, 2),
                error="No devices available for fit check",
            )

        try:
            from uniinfer.models.fitting import check_model_fit

            if active_device_str:
                device = next(
                    (d for d in hw_devices if d.device_string == active_device_str),
                    hw_devices[0],
                )
            else:
                device = select_best_device(preferred="auto", available=hw_devices)

            # Get alias info for param count (needed for alternatives)
            param_count = None
            try:
                from uniinfer.models.aliases import get_alias_info
                alias_info = get_alias_info(req.model_id)
                if alias_info:
                    param_count = alias_info.param_count_billions
            except Exception:
                pass

            report = check_model_fit(
                device=device,
                model_size_gb=size_gb,
                quantization=req.quantization,
                context_length=req.context_length,
                param_count_billions=param_count or 0.0,
            )

            alternatives = [
                FitAlternative(
                    quantization=alt.quantization,
                    estimated_size_gb=round(alt.estimated_size_gb, 2),
                    fits=alt.fits,
                )
                for alt in (report.alternatives or [])
            ]

            return DashboardFitCheckResponse(
                model_id=req.model_id,
                quantization=req.quantization,
                fits=report.fits,
                model_size_gb=round(report.model_size_gb, 2),
                available_memory_gb=round(report.available_memory_gb, 2),
                headroom_gb=round(report.headroom_gb, 2),
                overhead_gb=round(report.overhead_gb, 2),
                warnings=report.warnings,
                alternatives=alternatives,
                recommended_quantization=report.recommended_quantization,
                device_name=device.name,
            )

        except Exception as exc:
            return DashboardFitCheckResponse(
                model_id=req.model_id,
                quantization=req.quantization,
                model_size_gb=round(size_gb, 2),
                error=str(exc),
            )

    return router
