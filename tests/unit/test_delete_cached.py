"""Tests for delete_cached() in model registry."""

import json
import pytest
from pathlib import Path

from uniinfer.models.registry import delete_cached, get_cache_path, list_cached


@pytest.fixture
def temp_cache(tmp_path):
    """Create a temporary cache with a fake model."""
    cache_dir = tmp_path / "models"
    model_dir = cache_dir / "test--model" / "gguf"
    model_dir.mkdir(parents=True)

    # Create a fake GGUF file
    gguf_file = model_dir / "q4_k_m.gguf"
    gguf_file.write_bytes(b"\x00" * 1000)

    # Create metadata
    meta_dir = cache_dir / "test--model"
    meta_file = meta_dir / "metadata.json"
    meta_file.write_text(json.dumps({
        "model_id": "test/model",
        "source": "direct",
        "quantization": "q4_k_m",
    }))

    return str(tmp_path)


def test_delete_cached_removes_file(temp_cache):
    freed = delete_cached("test/model", "q4_k_m", cache_dir=temp_cache)
    assert freed == 1000

    # File should be gone
    path = get_cache_path("test/model", "q4_k_m", cache_dir=temp_cache)
    assert not path.exists()


def test_delete_cached_cleans_empty_dirs(temp_cache):
    delete_cached("test/model", "q4_k_m", cache_dir=temp_cache)

    cache_dir = Path(temp_cache) / "models"
    model_dir = cache_dir / "test--model"
    # Model dir should be cleaned up since it had only one file
    assert not model_dir.exists()


def test_delete_cached_file_not_found():
    with pytest.raises(FileNotFoundError):
        delete_cached("nonexistent/model", "q4_k_m")


def test_delete_cached_preserves_other_quants(temp_cache):
    # Add a second quantization
    cache_dir = Path(temp_cache) / "models"
    model_dir = cache_dir / "test--model" / "gguf"
    (model_dir / "q8_0.gguf").write_bytes(b"\x00" * 2000)

    # Delete only q4_k_m
    delete_cached("test/model", "q4_k_m", cache_dir=temp_cache)

    # q8_0 should still exist
    assert (model_dir / "q8_0.gguf").exists()
    # Model dir should still exist
    assert model_dir.parent.exists()
