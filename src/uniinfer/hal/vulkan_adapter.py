"""Vulkan device discovery via vulkaninfo CLI."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

from uniinfer.hal.interface import DeviceAdapter, DeviceInfo, DeviceType

logger = logging.getLogger(__name__)


class VulkanAdapter(DeviceAdapter):
    """Vulkan device adapter using vulkaninfo CLI.

    Parses vulkaninfo output to discover GPU devices.
    Gracefully fails if vulkaninfo is not installed.
    """

    def __init__(self) -> None:
        self._available = False
        self._devices: list[dict[str, str | int]] = []
        self._probe()

    def _probe(self) -> None:
        if shutil.which("vulkaninfo") is None:
            logger.debug("vulkaninfo not found — Vulkan discovery unavailable")
            return

        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                self._parse_summary(result.stdout)
            else:
                # Try without --summary flag (older versions)
                self._probe_full()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.debug("vulkaninfo probe failed: %s", exc)

    def _parse_summary(self, output: str) -> None:
        """Parse vulkaninfo --summary output for GPU info."""
        gpu_sections = re.split(r"GPU\d+:", output)

        for idx, section in enumerate(gpu_sections[1:], start=0):  # skip preamble
            name = "Unknown Vulkan GPU"
            total_memory = 0

            # Extract device name
            name_match = re.search(r"deviceName\s*=\s*(.+)", section)
            if name_match:
                name = name_match.group(1).strip()

            # Extract heap size (largest heap is typically device-local VRAM)
            heap_matches = re.findall(r"size\s*=\s*(\d+)", section)
            if heap_matches:
                total_memory = max(int(h) for h in heap_matches)

            # Skip integrated/software renderers that show as CPU
            device_type_match = re.search(r"deviceType\s*=\s*(\S+)", section)
            if device_type_match:
                dtype = device_type_match.group(1).upper()
                if "CPU" in dtype or "SOFTWARE" in dtype:
                    continue

            self._devices.append({
                "name": name,
                "device_id": idx,
                "total_memory": total_memory,
                "free_memory": total_memory,  # Vulkan doesn't report used memory easily
            })

        if self._devices:
            self._available = True
            logger.debug("Vulkan adapter found %d device(s)", len(self._devices))

    def _probe_full(self) -> None:
        """Fallback: parse full vulkaninfo output."""
        try:
            result = subprocess.run(
                ["vulkaninfo"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return

            output = result.stdout
            # Look for deviceName lines
            name_matches = re.findall(r"deviceName\s*=\s*(.+)", output)
            seen_names: set[str] = set()

            for idx, name_raw in enumerate(name_matches):
                name = name_raw.strip()
                if name in seen_names:
                    continue
                seen_names.add(name)

                self._devices.append({
                    "name": name,
                    "device_id": len(self._devices),
                    "total_memory": 0,
                    "free_memory": 0,
                })

            if self._devices:
                self._available = True
                logger.debug("Vulkan adapter found %d device(s) via full parse", len(self._devices))
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.debug("vulkaninfo full-mode probe failed: %s", exc)

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self._available or device_id >= len(self._devices):
            raise RuntimeError(f"Vulkan device {device_id} is not available")

        dev = self._devices[device_id]
        return DeviceInfo(
            name=str(dev["name"]),
            device_type=DeviceType.VULKAN,
            device_id=device_id,
            total_memory=int(dev["total_memory"]),
            free_memory=int(dev["free_memory"]),
        )

    def get_free_memory(self, device_id: int = 0) -> int:
        if not self._available or device_id >= len(self._devices):
            return 0
        return int(self._devices[device_id]["free_memory"])

    @property
    def is_available(self) -> bool:
        return self._available
