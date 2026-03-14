"""Tests for inference diagnostics."""

from __future__ import annotations

import time

import pytest

from uniinfer.engine.diagnostics import InferenceMetrics, SessionDiagnostics


class TestInferenceMetrics:
    def test_elapsed_seconds(self) -> None:
        metrics = InferenceMetrics(
            start_time=100.0,
            end_time=102.5,
            method="generate",
        )
        assert metrics.elapsed_seconds == pytest.approx(2.5)

    def test_tokens_per_second(self) -> None:
        metrics = InferenceMetrics(
            start_time=100.0,
            end_time=102.0,
            completion_tokens=50,
            method="generate",
        )
        assert metrics.tokens_per_second == pytest.approx(25.0)

    def test_zero_elapsed(self) -> None:
        metrics = InferenceMetrics(start_time=0.0, end_time=0.0)
        assert metrics.elapsed_seconds == 0.0
        assert metrics.tokens_per_second == 0.0

    def test_zero_tokens(self) -> None:
        metrics = InferenceMetrics(
            start_time=100.0,
            end_time=102.0,
            completion_tokens=0,
        )
        assert metrics.tokens_per_second == 0.0

    def test_total_tokens(self) -> None:
        metrics = InferenceMetrics(
            prompt_tokens=10,
            completion_tokens=50,
        )
        assert metrics.total_tokens == 60

    def test_to_dict(self) -> None:
        metrics = InferenceMetrics(
            start_time=100.0,
            end_time=102.0,
            prompt_tokens=10,
            completion_tokens=50,
            method="chat",
        )
        d = metrics.to_dict()
        assert d["method"] == "chat"
        assert d["elapsed_seconds"] == 2.0
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 60
        assert d["tokens_per_second"] == 25.0


class TestSessionDiagnostics:
    def test_empty_session(self) -> None:
        diag = SessionDiagnostics()
        assert diag.total_inferences == 0
        assert diag.total_tokens_generated == 0
        assert diag.total_inference_time == 0.0
        assert diag.average_tokens_per_second == 0.0
        assert diag.peak_tokens_per_second == 0.0

    def test_record_single_inference(self) -> None:
        diag = SessionDiagnostics()
        metrics = InferenceMetrics(
            start_time=100.0,
            end_time=102.0,
            prompt_tokens=10,
            completion_tokens=50,
            method="generate",
        )
        diag.record(metrics)

        assert diag.total_inferences == 1
        assert diag.total_tokens_generated == 50
        assert diag.total_inference_time == pytest.approx(2.0)
        assert diag.average_tokens_per_second == pytest.approx(25.0)
        assert diag.peak_tokens_per_second == pytest.approx(25.0)

    def test_record_multiple_inferences(self) -> None:
        diag = SessionDiagnostics()

        # First: 50 tokens in 2 seconds = 25 tok/s
        diag.record(InferenceMetrics(
            start_time=100.0, end_time=102.0,
            completion_tokens=50, method="generate",
        ))

        # Second: 100 tokens in 2 seconds = 50 tok/s
        diag.record(InferenceMetrics(
            start_time=200.0, end_time=202.0,
            completion_tokens=100, method="chat",
        ))

        assert diag.total_inferences == 2
        assert diag.total_tokens_generated == 150
        assert diag.total_inference_time == pytest.approx(4.0)
        assert diag.average_tokens_per_second == pytest.approx(37.5)
        assert diag.peak_tokens_per_second == pytest.approx(50.0)

    def test_model_load_time(self) -> None:
        diag = SessionDiagnostics(model_load_time=3.5)
        assert diag.model_load_time == 3.5

    def test_to_dict(self) -> None:
        diag = SessionDiagnostics(model_load_time=1.5)
        diag.record(InferenceMetrics(
            start_time=100.0, end_time=101.0,
            prompt_tokens=5, completion_tokens=20,
            method="chat",
        ))

        d = diag.to_dict()
        assert d["model_load_time_seconds"] == 1.5
        assert d["total_inferences"] == 1
        assert d["total_tokens_generated"] == 20
        assert d["average_tokens_per_second"] == 20.0
        assert d["peak_tokens_per_second"] == 20.0
        assert d["last_inference"] is not None
        assert d["last_inference"]["method"] == "chat"

    def test_to_dict_empty(self) -> None:
        diag = SessionDiagnostics()
        d = diag.to_dict()
        assert d["last_inference"] is None
        assert d["total_inferences"] == 0
