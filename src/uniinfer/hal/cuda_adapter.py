"""CUDA device discovery using pynvml."""

from __future__ import annotations

import logging
from typing import Optional

from uniinfer.hal.interface import DeviceAdapter, DeviceInfo, DeviceType

logger = logging.getLogger(__name__)


class CudaAdapter(DeviceAdapter):
    """CUDA device adapter using pynvml for GPU discovery.

    Gracefully handles the case where pynvml is not installed
    or NVIDIA drivers are not available.
    """

    def __init__(self) -> None:
        self._available = False
        self._nvml_initialized = False
        self._pynvml: Optional[object] = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_initialized = True
            self._available = True
            logger.debug("CUDA adapter initialized via pynvml")
        except ImportError:
            logger.debug("pynvml not installed — CUDA discovery unavailable")
        except Exception as exc:
            logger.debug("pynvml init failed: %s", exc)

    def get_device_count(self) -> int:
        if not self._available or self._pynvml is None:
            return 0
        try:
            import pynvml  # type: ignore[import-untyped]

            return pynvml.nvmlDeviceGetCount()  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("Failed to get CUDA device count: %s", exc)
            return 0

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self._available or self._pynvml is None:
            raise RuntimeError("CUDA is not available")

        import pynvml  # type: ignore[import-untyped]

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Get compute capability
        compute_cap: Optional[tuple[int, int]] = None
        try:
            major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if isinstance(major, tuple):
                compute_cap = (major[0], major[1])
            else:
                minor = 0
                compute_cap = (major, minor)
        except Exception:
            pass

        # Get driver version
        extra: dict[str, str] = {}
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8")
            extra["driver_version"] = driver
        except Exception:
            pass

        return DeviceInfo(
            name=name,
            device_type=DeviceType.CUDA,
            device_id=device_id,
            total_memory=mem_info.total,
            free_memory=mem_info.free,
            compute_capability=compute_cap,
            extra=extra,
        )

    def get_free_memory(self, device_id: int = 0) -> int:
        if not self._available or self._pynvml is None:
            return 0
        try:
            import pynvml  # type: ignore[import-untyped]

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.free  # type: ignore[no-any-return]
        except Exception:
            return 0

    @property
    def is_available(self) -> bool:
        return self._available
