"""Shared test fixtures for UniInfer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from uniinfer.hal.interface import DeviceInfo, DeviceType


@pytest.fixture
def cpu_device() -> DeviceInfo:
    """A mock CPU device with 16 GB RAM, 12 GB free."""
    return DeviceInfo(
        name="Test CPU",
        device_type=DeviceType.CPU,
        device_id=0,
        total_memory=16 * 1024**3,
        free_memory=12 * 1024**3,
        extra={"cores": "8", "threads": "16"},
    )


@pytest.fixture
def cuda_device_24gb() -> DeviceInfo:
    """A mock CUDA device with 24 GB VRAM, ~24 GB free."""
    return DeviceInfo(
        name="NVIDIA RTX 4090",
        device_type=DeviceType.CUDA,
        device_id=0,
        total_memory=24 * 1024**3,
        free_memory=24 * 1024**3,
        compute_capability=(8, 9),
        extra={"driver_version": "550.0"},
    )


@pytest.fixture
def cuda_device_8gb() -> DeviceInfo:
    """A mock CUDA device with 8 GB VRAM."""
    return DeviceInfo(
        name="NVIDIA RTX 3060",
        device_type=DeviceType.CUDA,
        device_id=0,
        total_memory=8 * 1024**3,
        free_memory=7 * 1024**3,
        compute_capability=(8, 6),
    )


@pytest.fixture
def cuda_device_4gb() -> DeviceInfo:
    """A mock CUDA device with 4 GB VRAM."""
    return DeviceInfo(
        name="NVIDIA GTX 1650",
        device_type=DeviceType.CUDA,
        device_id=0,
        total_memory=4 * 1024**3,
        free_memory=3 * 1024**3,
        compute_capability=(7, 5),
    )


@pytest.fixture
def rocm_device() -> DeviceInfo:
    """A mock ROCm device with 16 GB VRAM."""
    return DeviceInfo(
        name="AMD Radeon RX 7800 XT",
        device_type=DeviceType.ROCM,
        device_id=0,
        total_memory=16 * 1024**3,
        free_memory=14 * 1024**3,
    )


@pytest.fixture
def vulkan_device() -> DeviceInfo:
    """A mock Vulkan device with 8 GB VRAM."""
    return DeviceInfo(
        name="Vulkan GPU",
        device_type=DeviceType.VULKAN,
        device_id=0,
        total_memory=8 * 1024**3,
        free_memory=7 * 1024**3,
    )
