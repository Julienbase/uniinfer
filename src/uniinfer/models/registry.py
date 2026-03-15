"""Model registry — download, cache, and catalog management."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from uniinfer.models.quantization import get_gguf_search_patterns

if TYPE_CHECKING:
    from uniinfer.hal.interface import DeviceInfo

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(os.path.expanduser("~")) / ".uniinfer" / "cache" / "models"


@dataclass
class CachedModel:
    """Metadata for a cached model."""

    model_id: str
    gguf_path: str  # path to model file or directory
    quantization: str
    file_size: int  # bytes
    source: str  # "direct", "gguf_variant", "onnx", "safetensors"
    format: str = "gguf"  # "gguf", "onnx", "safetensors"
    extra: dict[str, str] = field(default_factory=dict)


def _sanitize_model_id(model_id: str) -> str:
    """Convert HuggingFace model ID to a safe directory name."""
    return model_id.replace("/", "--")


def _cache_dir_for_model(model_id: str, base_dir: Path, fmt: str = "gguf") -> Path:
    """Return the cache directory for a given model and format."""
    return base_dir / _sanitize_model_id(model_id) / fmt


def _metadata_path(model_id: str, base_dir: Path) -> Path:
    return base_dir / _sanitize_model_id(model_id) / "metadata.json"


def detect_repo_format(model_id: str) -> str:
    """Detect the model format of a HuggingFace repo.

    Checks the repo's file listing for format indicators:
    - .gguf files → "gguf"
    - .onnx files → "onnx"
    - .safetensors files → "safetensors"

    Args:
        model_id: HuggingFace model ID.

    Returns:
        Format string: "gguf", "onnx", or "safetensors".
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            files = list(api.list_repo_files(model_id))
        except Exception:
            return "gguf"  # default assumption

        has_gguf = any(f.endswith(".gguf") for f in files)
        has_onnx = any(f.endswith(".onnx") for f in files)
        has_safetensors = any(f.endswith(".safetensors") for f in files)

        if has_gguf:
            return "gguf"
        if has_onnx:
            return "onnx"
        if has_safetensors:
            return "safetensors"
        return "gguf"  # fallback

    except ImportError:
        return "gguf"


def get_cache_path(
    model_id: str,
    quantization: str,
    cache_dir: Optional[str] = None,
) -> Path:
    """Return expected path to a cached GGUF file.

    Args:
        model_id: HuggingFace model ID.
        quantization: Quantization level (f16, q8_0, q4_k_m).
        cache_dir: Base cache directory. Defaults to ~/.uniinfer/cache/models.

    Returns:
        Path where the GGUF file would be cached.
    """
    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR
    model_dir = _cache_dir_for_model(model_id, base, "gguf")
    return model_dir / f"{quantization}.gguf"


def get_model_cache_path(
    model_id: str,
    fmt: str = "gguf",
    quantization: str = "q4_k_m",
    cache_dir: Optional[str] = None,
) -> Path:
    """Return expected cache path for any model format.

    Args:
        model_id: HuggingFace model ID.
        fmt: Model format ("gguf", "onnx", "safetensors").
        quantization: Quantization level (only used for GGUF).
        cache_dir: Base cache directory.

    Returns:
        Path to the cached model file (GGUF) or directory (ONNX/SafeTensors).
    """
    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR
    if fmt == "gguf":
        return _cache_dir_for_model(model_id, base, "gguf") / f"{quantization}.gguf"
    return _cache_dir_for_model(model_id, base, fmt)


def is_cached(
    model_id: str,
    quantization: str,
    cache_dir: Optional[str] = None,
) -> bool:
    """Check if a model is already cached.

    Checks for GGUF file first, then ONNX/SafeTensors directories.

    Args:
        model_id: HuggingFace model ID.
        quantization: Quantization level (used for GGUF).
        cache_dir: Base cache directory.

    Returns:
        True if the model is cached in any format.
    """
    # Check GGUF
    path = get_cache_path(model_id, quantization, cache_dir)
    if path.exists() and path.stat().st_size > 0:
        return True
    # Check ONNX directory
    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR
    onnx_dir = _cache_dir_for_model(model_id, base, "onnx")
    if onnx_dir.exists() and any(onnx_dir.rglob("*.onnx")):
        return True
    # Check SafeTensors directory
    st_dir = _cache_dir_for_model(model_id, base, "safetensors")
    if st_dir.exists() and any(st_dir.rglob("*.safetensors")):
        return True
    return False


def get_cached_path(
    model_id: str,
    quantization: str,
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """Return the actual cached path for a model, checking all formats.

    Args:
        model_id: HuggingFace model ID.
        quantization: Quantization level (used for GGUF).
        cache_dir: Base cache directory.

    Returns:
        Path to the cached model (file for GGUF, directory for ONNX/SafeTensors),
        or None if not cached.
    """
    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR

    # Check GGUF first
    gguf_path = get_cache_path(model_id, quantization, cache_dir)
    if gguf_path.exists() and gguf_path.stat().st_size > 0:
        return gguf_path

    # Check ONNX directory
    onnx_dir = _cache_dir_for_model(model_id, base, "onnx")
    if onnx_dir.exists():
        onnx_files = list(onnx_dir.rglob("*.onnx"))
        if onnx_files:
            # Prefer models that are self-contained or have their _data companion
            usable = [
                f for f in onnx_files
                if f.stat().st_size >= 10 * 1024 * 1024
                or Path(str(f) + "_data").exists()
            ]
            if usable:
                return usable[0]
            return onnx_files[0]

    # Check SafeTensors directory
    st_dir = _cache_dir_for_model(model_id, base, "safetensors")
    if st_dir.exists() and any(st_dir.rglob("*.safetensors")):
        return st_dir  # return directory for transformers backend

    return None


def _dir_total_size(directory: Path) -> int:
    """Calculate total size of all files in a directory."""
    total = 0
    for f in directory.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def list_cached(cache_dir: Optional[str] = None) -> list[CachedModel]:
    """List all cached models (GGUF, ONNX, and SafeTensors).

    Args:
        cache_dir: Base cache directory.

    Returns:
        List of CachedModel entries.
    """
    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR
    cached: list[CachedModel] = []

    if not base.exists():
        return cached

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        # Skip HF cache directory
        if model_dir.name == "_hf_cache":
            continue

        model_id = model_dir.name.replace("--", "/")
        metadata_file = model_dir / "metadata.json"
        meta: dict = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        source = meta.get("source", "unknown")
        fmt = meta.get("format", "gguf")

        # Check GGUF directory
        gguf_dir = model_dir / "gguf"
        if gguf_dir.exists():
            for gguf_file in sorted(gguf_dir.glob("*.gguf")):
                cached.append(CachedModel(
                    model_id=model_id,
                    gguf_path=str(gguf_file),
                    quantization=gguf_file.stem,
                    file_size=gguf_file.stat().st_size,
                    source=source,
                    format="gguf",
                ))

        # Check ONNX directory
        onnx_dir = model_dir / "onnx"
        if onnx_dir.exists():
            onnx_files = list(onnx_dir.rglob("*.onnx"))
            # Prefer models that are self-contained or have their _data companion
            usable = [
                f for f in onnx_files
                if f.stat().st_size >= 10 * 1024 * 1024
                or Path(str(f) + "_data").exists()
            ]
            best = usable[0] if usable else (onnx_files[0] if onnx_files else None)
            if best:
                cached.append(CachedModel(
                    model_id=model_id,
                    gguf_path=str(best),
                    quantization="native",
                    file_size=_dir_total_size(onnx_dir),
                    source=source,
                    format="onnx",
                ))

        # Check SafeTensors directory
        st_dir = model_dir / "safetensors"
        if st_dir.exists() and any(st_dir.rglob("*.safetensors")):
            cached.append(CachedModel(
                model_id=model_id,
                gguf_path=str(st_dir),
                quantization="native",
                file_size=_dir_total_size(st_dir),
                source=source,
                format="safetensors",
            ))

    return cached


def _list_gguf_files_with_sizes(model_id: str) -> list[tuple[str, int]]:
    """List all GGUF files in a HuggingFace repo with their sizes.

    Args:
        model_id: HuggingFace model ID.

    Returns:
        List of (filename, size_bytes) tuples for GGUF files.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            repo_info = api.repo_info(model_id, files_metadata=True)
        except Exception:
            return []

        results = []
        for sibling in repo_info.siblings or []:
            if sibling.rfilename.endswith(".gguf"):
                size = getattr(sibling, "size", None) or 0
                results.append((sibling.rfilename, size))
        return results

    except ImportError:
        logger.error("huggingface_hub is required for model discovery")
        return []


def query_model_size_from_hf(
    model_id: str, quantization: str = "q4_k_m",
) -> Optional[tuple[str, float]]:
    """Query HuggingFace for actual GGUF file size without downloading.

    Searches the model repo (and GGUF variant repos) for a matching file
    and returns its size in GB.

    Args:
        model_id: HuggingFace model ID.
        quantization: Desired quantization level.

    Returns:
        Tuple of (filename, size_gb) or None if no GGUF files found.
    """
    # Try the original repo first
    gguf_files = _list_gguf_files_with_sizes(model_id)

    # If no GGUF files, try variant repos
    if not gguf_files:
        variant = _find_gguf_variant_repo(model_id)
        if variant:
            gguf_files = _list_gguf_files_with_sizes(variant)

    if not gguf_files:
        return None

    # Try to find matching quantization
    patterns = get_gguf_search_patterns(quantization)
    for filename, size_bytes in gguf_files:
        for pattern in patterns:
            if pattern.lower() in filename.lower():
                return (filename, size_bytes / (1024**3))

    # Fallback to first GGUF file
    filename, size_bytes = gguf_files[0]
    return (filename, size_bytes / (1024**3))


def query_any_model_size_from_hf(
    model_id: str, quantization: str = "q4_k_m",
) -> Optional[tuple[str, float, str]]:
    """Query HuggingFace for model size, supporting all formats.

    Tries GGUF first, then checks for ONNX or SafeTensors files and
    returns the total repo size for those formats.

    Args:
        model_id: HuggingFace model ID.
        quantization: Desired quantization level (used for GGUF).

    Returns:
        Tuple of (description, size_gb, format) or None if repo not found.
    """
    # Try GGUF first
    gguf_result = query_model_size_from_hf(model_id, quantization)
    if gguf_result is not None:
        filename, size_gb = gguf_result
        return (filename, size_gb, "gguf")

    # Try ONNX/SafeTensors by checking repo files
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            repo_info = api.repo_info(model_id, files_metadata=True)
        except Exception:
            return None

        total_size = 0
        fmt = ""
        file_count = 0
        for sibling in repo_info.siblings or []:
            size = getattr(sibling, "size", None) or 0
            fname = sibling.rfilename
            if fname.endswith(".onnx"):
                fmt = "onnx"
                total_size += size
                file_count += 1
            elif fname.endswith(".safetensors"):
                fmt = "safetensors"
                total_size += size
                file_count += 1
            elif fname.endswith((".json", ".txt", ".model", ".tiktoken")):
                # Include tokenizer and config files in total
                total_size += size

        if fmt and total_size > 0:
            desc = f"{file_count} {fmt.upper()} file{'s' if file_count > 1 else ''}"
            return (desc, total_size / (1024**3), fmt)

        return None

    except ImportError:
        return None


def _search_gguf_in_repo(model_id: str, quantization: str) -> Optional[str]:
    """Search for GGUF files within a HuggingFace model repo.

    Args:
        model_id: HuggingFace model ID.
        quantization: Desired quantization level.

    Returns:
        Filename of the matching GGUF file, or None.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            files = api.list_repo_files(model_id)
        except Exception:
            return None

        gguf_files = [f for f in files if f.endswith(".gguf")]
        if not gguf_files:
            return None

        # Try to find matching quantization
        patterns = get_gguf_search_patterns(quantization)
        for gguf_file in gguf_files:
            for pattern in patterns:
                if pattern.lower() in gguf_file.lower():
                    logger.info("Found matching GGUF: %s in %s", gguf_file, model_id)
                    return gguf_file

        # If no quantization match, return the first GGUF file
        logger.info("No exact quantization match, using: %s", gguf_files[0])
        return gguf_files[0]

    except ImportError:
        logger.error("huggingface_hub is required for model discovery")
        return None


def _find_gguf_variant_repo(model_id: str) -> Optional[str]:
    """Search for well-known GGUF variant repos on HuggingFace.

    Many models have community-uploaded GGUF variants. Common patterns:
    - TheBloke/<model>-GGUF
    - bartowski/<model>-GGUF
    - <user>/<model>-GGUF

    Args:
        model_id: Original HuggingFace model ID.

    Returns:
        GGUF variant repo ID, or None.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Extract model name from the ID
        model_name = model_id.split("/")[-1]

        # Search for GGUF variants
        search_query = f"{model_name} GGUF"
        try:
            results = list(api.list_models(search=search_query, limit=10))
        except Exception:
            return None

        # Filter for repos that contain GGUF files
        for result in results:
            repo_id = result.id if hasattr(result, "id") else str(result)
            if "gguf" in repo_id.lower() or "GGUF" in repo_id:
                # Verify it actually has GGUF files
                try:
                    files = api.list_repo_files(repo_id)
                    if any(f.endswith(".gguf") for f in files):
                        logger.info("Found GGUF variant repo: %s", repo_id)
                        return repo_id
                except Exception:
                    continue

        return None

    except ImportError:
        logger.error("huggingface_hub is required for GGUF variant search")
        return None


def delete_cached(
    model_id: str,
    quantization: str,
    cache_dir: Optional[str] = None,
    fmt: str = "gguf",
) -> int:
    """Delete a cached model file or directory.

    Args:
        model_id: HuggingFace model ID.
        quantization: Quantization level (used for GGUF).
        cache_dir: Base cache directory.
        fmt: Model format ("gguf", "onnx", "safetensors").

    Returns:
        Number of bytes freed.

    Raises:
        FileNotFoundError: If the model is not cached.
    """
    import shutil

    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR

    if fmt in ("onnx", "safetensors"):
        # Delete the entire format directory
        fmt_dir = _cache_dir_for_model(model_id, base, fmt)
        if not fmt_dir.exists():
            raise FileNotFoundError(
                f"No cached {fmt} model found for '{model_id}'"
            )
        size = _dir_total_size(fmt_dir)
        shutil.rmtree(str(fmt_dir))
    else:
        # Delete single GGUF file
        path = get_cache_path(model_id, quantization, cache_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached model found for '{model_id}' at quantization '{quantization}'"
            )
        size = path.stat().st_size
        path.unlink()

        # Clean up empty gguf directory
        gguf_dir = path.parent
        if gguf_dir.exists() and not any(gguf_dir.iterdir()):
            gguf_dir.rmdir()

    # Clean up model directory if empty
    model_dir = base / _sanitize_model_id(model_id)
    meta = model_dir / "metadata.json"
    if meta.exists():
        # Check if any format dirs remain
        remaining = [d for d in model_dir.iterdir() if d.is_dir()]
        if not remaining:
            meta.unlink()
    if model_dir.exists() and not any(model_dir.iterdir()):
        model_dir.rmdir()

    logger.info("Deleted cached model: %s (%s/%s), freed %d bytes", model_id, fmt, quantization, size)
    return size


def _download_snapshot(
    model_id: str,
    fmt: str,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download a full HuggingFace repo (ONNX or SafeTensors) to cache.

    Uses snapshot_download to pull the entire repo, then symlinks or copies
    to our cache structure.

    Args:
        model_id: HuggingFace model ID.
        fmt: Format string ("onnx" or "safetensors").
        cache_dir: Base cache directory.

    Returns:
        Path to the cached model directory (for SafeTensors) or .onnx file (for ONNX).
    """
    from huggingface_hub import snapshot_download

    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR
    target_dir = _cache_dir_for_model(model_id, base, fmt)

    if target_dir.exists():
        # Already cached — find the right file/dir
        if fmt == "onnx":
            onnx_files = list(target_dir.rglob("*.onnx"))
            if onnx_files:
                return onnx_files[0]
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s model '%s' from HuggingFace...", fmt.upper(), model_id)

    # Download the full repo snapshot
    snapshot_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(base / "_hf_cache"),
        local_dir=str(target_dir),
    )

    # Save metadata
    metadata_path = _metadata_path(model_id, base)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": model_id,
        "source_repo": model_id,
        "format": fmt,
        "quantization": "native",
        "source": "direct",
    }
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write metadata: %s", exc)

    # For ONNX, return the path to the .onnx file
    if fmt == "onnx":
        onnx_files = list(target_dir.rglob("*.onnx"))
        if onnx_files:
            # Prefer model.onnx or the largest .onnx file
            for f in onnx_files:
                if f.name == "model.onnx":
                    logger.info("ONNX model cached at: %s", f)
                    return f
            largest = max(onnx_files, key=lambda p: p.stat().st_size)
            logger.info("ONNX model cached at: %s", largest)
            return largest
        raise RuntimeError(
            f"Downloaded repo '{model_id}' but no .onnx files found in snapshot"
        )

    logger.info("Model cached at: %s", target_dir)
    return target_dir


def download_model(
    model_id: str,
    quantization: str = "q4_k_m",
    cache_dir: Optional[str] = None,
    device: Optional["DeviceInfo"] = None,
    param_count_billions: float = 0.0,
) -> Path:
    """Download a model and return the path to the model file or directory.

    Supports GGUF, ONNX, and SafeTensors formats. Auto-detects the repo
    format and downloads accordingly.

    Strategy:
    1. Check if already cached (any format).
    2. Detect repo format on HuggingFace.
    3. For GGUF: search for GGUF files, try variant repos.
    4. For ONNX/SafeTensors: download full repo via snapshot_download.

    Args:
        model_id: HuggingFace model ID.
        quantization: Desired quantization level (GGUF only).
        cache_dir: Base cache directory.
        device: Target device info for pre-download fit validation.
        param_count_billions: Estimated param count for fit check. 0 = skip.

    Returns:
        Path to the downloaded model file (GGUF/ONNX) or directory (SafeTensors).

    Raises:
        RuntimeError: If no supported model files can be found.
        ModelTooLargeError: If the model won't fit on the target device.
    """
    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR

    # 1. Check cache (all formats)
    existing = get_cached_path(model_id, quantization, cache_dir)
    if existing is not None:
        logger.info("Model already cached: %s", existing)
        return existing

    # 2. Detect repo format
    fmt = detect_repo_format(model_id)
    logger.info("Detected repo format for '%s': %s", model_id, fmt)

    # 3. For ONNX or SafeTensors, use snapshot download
    if fmt in ("onnx", "safetensors"):
        return _download_snapshot(model_id, fmt, cache_dir)

    # 4. GGUF path — pre-download fit check
    if device is not None and param_count_billions > 0:
        from uniinfer.models.fitting import (
            ModelTooLargeError,
            check_model_fit,
            estimate_model_size_gb,
        )

        est_size = estimate_model_size_gb(param_count_billions, quantization)
        report = check_model_fit(
            device=device,
            model_size_gb=est_size,
            quantization=quantization,
            param_count_billions=param_count_billions,
        )
        if not report.fits:
            alt_msg = ""
            if report.recommended_quantization != quantization:
                alt_msg = (
                    f"\n  Recommendation: try '{report.recommended_quantization}' "
                    f"quantization instead."
                )
            if report.recommended_context_length < 4096:
                alt_msg += (
                    f"\n  Also consider reducing context to "
                    f"{report.recommended_context_length} tokens."
                )
            raise ModelTooLargeError(
                f"Model '{model_id}' (~{est_size:.1f} GB at {quantization}) "
                f"won't fit on {device.name} "
                f"({device.free_memory_gb:.1f} GB free).{alt_msg}",
                fit_report=report,
            )
        logger.info(
            "Pre-download fit check passed: ~%.1f GB model, %.1f GB headroom",
            est_size,
            report.headroom_gb,
        )

    # Search for GGUF in the original repo
    from huggingface_hub import hf_hub_download

    gguf_filename = _search_gguf_in_repo(model_id, quantization)
    source_repo = model_id

    # If not found, search for GGUF variant repos
    if gguf_filename is None:
        variant_repo = _find_gguf_variant_repo(model_id)
        if variant_repo:
            gguf_filename = _search_gguf_in_repo(variant_repo, quantization)
            if gguf_filename:
                source_repo = variant_repo

    # If still no GGUF, raise error
    if gguf_filename is None:
        raise RuntimeError(
            f"No supported model files found for '{model_id}'.\n\n"
            f"Options:\n"
            f"  1. Use a model that has GGUF, ONNX, or SafeTensors files\n"
            f"  2. Provide a direct path to a local model file\n"
            f"  3. Search for '{model_id.split('/')[-1]}' variants on huggingface.co\n"
        )

    # Download the GGUF file
    logger.info("Downloading %s from %s ...", gguf_filename, source_repo)

    cached_path = get_cache_path(model_id, quantization, cache_dir)
    cached_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=source_repo,
        filename=gguf_filename,
        cache_dir=str(base / "_hf_cache"),
    )

    # Symlink or copy to our cache structure
    downloaded = Path(downloaded_path)
    if not cached_path.exists():
        try:
            cached_path.symlink_to(downloaded)
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(str(downloaded), str(cached_path))

    # Save metadata
    metadata_path = _metadata_path(model_id, base)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": model_id,
        "source_repo": source_repo,
        "gguf_filename": gguf_filename,
        "quantization": quantization,
        "format": "gguf",
        "source": "gguf_variant" if source_repo != model_id else "direct",
    }
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write metadata: %s", exc)

    logger.info("Model cached at: %s", cached_path)
    return cached_path
