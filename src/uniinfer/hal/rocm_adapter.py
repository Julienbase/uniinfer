"""ROCm device discovery via rocm-smi CLI."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

from uniinfer.hal.interface import DeviceAdapter, DeviceInfo, DeviceType

logger = logging.getLogger(__name__)


class RocmAdapter(DeviceAdapter):
    """ROCm device adapter using rocm-smi for AMD GPU discovery.

    Parses output from the rocm-smi command-line tool.
    Gracefully fails if rocm-smi is not installed.
    """

    def __init__(self) -> None:
        self._available = False
        self._devices: list[dict[str, str | int]] = []
        self._probe()

    def _probe(self) -> None:
        if shutil.which("rocm-smi") is None:
            logger.debug("rocm-smi not found — ROCm discovery unavailable")
            return

        try:
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._parse_json_output(result.stdout)
            else:
                # Fallback to non-JSON output
                self._probe_text_mode()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.debug("rocm-smi probe failed: %s", exc)

    def _parse_json_output(self, output: str) -> None:
        import json

        try:
            data = json.loads(output)
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                device_id_match = re.search(r"(\d+)", key)
                device_id = int(device_id_match.group(1)) if device_id_match else len(self._devices)

                name = str(val.get("Card series", val.get("card_series", f"AMD GPU {device_id}")))
                total_vram = int(val.get("VRAM Total Memory (B)", val.get("vram_total", 0)))
                used_vram = int(val.get("VRAM Total Used Memory (B)", val.get("vram_used", 0)))

                self._devices.append({
                    "name": name,
                    "device_id": device_id,
                    "total_memory": total_vram,
                    "free_memory": max(0, total_vram - used_vram),
                })
            if self._devices:
                self._available = True
                logger.debug("ROCm adapter found %d device(s)", len(self._devices))
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.debug("Failed to parse rocm-smi JSON: %s", exc)
            self._probe_text_mode()

    def _probe_text_mode(self) -> None:
        """Fallback: count GPUs from rocm-smi text output."""
        try:
            result = subprocess.run(
                ["rocm-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return

            # Parse the table output for GPU lines
            gpu_lines = [
                line for line in result.stdout.splitlines()
                if re.match(r"^\s*\d+\s+", line) and "GPU" not in line.upper().split()[0:1]
            ]

            for idx, _line in enumerate(gpu_lines):
                self._devices.append({
                    "name": f"AMD GPU {idx}",
                    "device_id": idx,
                    "total_memory": 0,
                    "free_memory": 0,
                })

            if self._devices:
                self._available = True
                logger.debug("ROCm adapter found %d device(s) via text mode", len(self._devices))
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.debug("rocm-smi text-mode probe failed: %s", exc)

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self._available or device_id >= len(self._devices):
            raise RuntimeError(f"ROCm device {device_id} is not available")

        dev = self._devices[device_id]
        return DeviceInfo(
            name=str(dev["name"]),
            device_type=DeviceType.ROCM,
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
