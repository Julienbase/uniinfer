"""Tests for dashboard Pydantic schemas."""

import pytest

from uniinfer.api.dashboard_schemas import (
    AliasListResponse,
    AliasResponse,
    CachedModelResponse,
    CachedModelsListResponse,
    DeviceListResponse,
    DeviceResponse,
    DiagnosticsInfo,
    FallbackEvent,
    FallbackInfo,
    FitInfo,
    LastInference,
    ModelDeleteRequest,
    ModelDeleteResponse,
    ModelDownloadRequest,
    ModelSizeResponse,
    SchedulerInfo,
    StatusResponse,
)


class TestStatusResponse:
    def test_default_values(self):
        status = StatusResponse()
        assert status.model == ""
        assert status.loaded is False
        assert status.uptime_seconds == 0.0
        assert status.fit is None
        assert status.fallback is None

    def test_with_full_data(self):
        status = StatusResponse(
            model="test-model",
            device="cuda:0",
            device_name="RTX 3060",
            quantization="q4_k_m",
            context_length=4096,
            backend="llama_cpp",
            loaded=True,
            device_memory_total_gb=12.0,
            device_memory_free_gb=8.0,
            uptime_seconds=3600.0,
            fit=FitInfo(fits=True, model_size_gb=4.0, headroom_gb=3.5, warnings=[]),
            diagnostics=DiagnosticsInfo(
                total_inferences=10,
                average_tokens_per_second=50.0,
            ),
            scheduler=SchedulerInfo(queue_depth=0, is_processing=False),
        )
        assert status.loaded is True
        assert status.device_name == "RTX 3060"
        assert status.fit.fits is True
        assert status.diagnostics.total_inferences == 10

    def test_fit_info_with_warnings(self):
        fit = FitInfo(
            fits=False,
            model_size_gb=20.0,
            headroom_gb=-8.0,
            warnings=["Model too large"],
        )
        assert fit.fits is False
        assert len(fit.warnings) == 1


class TestFallbackInfo:
    def test_fallback_event(self):
        event = FallbackEvent(**{
            "from": "cuda:0",
            "to": "cpu",
            "reason": "OOM",
            "success": True,
        })
        assert event.from_device == "cuda:0"
        assert event.to_device == "cpu"

    def test_fallback_info_with_events(self):
        info = FallbackInfo(
            fell_back=True,
            summary="Fell back from cuda:0 to cpu",
            events=[
                FallbackEvent(**{
                    "from": "cuda:0",
                    "to": "cpu",
                    "reason": "OOM",
                    "success": True,
                })
            ],
        )
        assert info.fell_back is True
        assert len(info.events) == 1


class TestDiagnosticsInfo:
    def test_defaults(self):
        diag = DiagnosticsInfo()
        assert diag.total_inferences == 0
        assert diag.last_inference is None

    def test_with_last_inference(self):
        diag = DiagnosticsInfo(
            total_inferences=5,
            last_inference=LastInference(
                method="chat",
                elapsed_seconds=1.5,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                tokens_per_second=33.3,
            ),
        )
        assert diag.last_inference.method == "chat"
        assert diag.last_inference.total_tokens == 150


class TestDeviceResponse:
    def test_device_response(self):
        dev = DeviceResponse(
            name="RTX 3060",
            device_string="cuda:0",
            device_type="cuda",
            total_memory_gb=12.0,
            free_memory_gb=8.0,
            is_active=True,
        )
        assert dev.is_active is True
        assert dev.device_type == "cuda"

    def test_device_list_response(self):
        resp = DeviceListResponse(devices=[
            DeviceResponse(
                name="CPU",
                device_string="cpu",
                device_type="cpu",
                total_memory_gb=32.0,
                free_memory_gb=28.0,
            )
        ])
        assert len(resp.devices) == 1


class TestCachedModelResponse:
    def test_cached_model(self):
        model = CachedModelResponse(
            model_id="TheBloke/Mistral-7B-GGUF",
            quantization="q4_k_m",
            file_size_bytes=4_500_000_000,
            file_size_gb=4.19,
            source="gguf_variant",
            gguf_path="/path/to/model.gguf",
            is_loaded=True,
        )
        assert model.is_loaded is True
        assert model.file_size_gb == 4.19


class TestAliasResponse:
    def test_alias(self):
        alias = AliasResponse(
            alias="mistral-7b",
            display_name="Mistral 7B Instruct",
            repo_id="TheBloke/Mistral-7B-GGUF",
            param_count_billions=7.24,
            default_quant="q4_k_m",
            default_context_length=4096,
            is_cached=False,
        )
        assert alias.param_count_billions == 7.24
        assert alias.is_cached is False


class TestModelSizeResponse:
    def test_successful_response(self):
        resp = ModelSizeResponse(
            model_id="test",
            quantization="q4_k_m",
            filename="model.gguf",
            size_gb=4.7,
            fits=True,
            headroom_gb=3.5,
        )
        assert resp.fits is True
        assert resp.error is None

    def test_error_response(self):
        resp = ModelSizeResponse(
            model_id="test",
            quantization="q4_k_m",
            error="Not found",
        )
        assert resp.error == "Not found"
        assert resp.size_gb is None


class TestRequestSchemas:
    def test_model_delete_request(self):
        req = ModelDeleteRequest(model_id="test-model", quantization="q4_k_m")
        assert req.model_id == "test-model"

    def test_model_download_request_default_quant(self):
        req = ModelDownloadRequest(model_id="test-model")
        assert req.quantization == "q4_k_m"

    def test_model_delete_response(self):
        resp = ModelDeleteResponse(deleted=True, freed_bytes=5_000_000_000)
        assert resp.deleted is True
        assert resp.freed_bytes == 5_000_000_000
