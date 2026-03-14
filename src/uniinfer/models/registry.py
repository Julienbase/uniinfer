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
    gguf_path: str
    quantization: str
    file_size: int  # bytes
    source: str  # "direct" or "gguf_variant"
    extra: dict[str, str] = field(default_factory=dict)


def _sanitize_model_id(model_id: str) -> str:
    """Convert HuggingFace model ID to a safe directory name."""
    return model_id.replace("/", "--")


def _cache_dir_for_model(model_id: str, base_dir: Path) -> Path:
    """Return the cache directory for a given model."""
    return base_dir / _sanitize_model_id(model_id) / "gguf"


def _metadata_path(model_id: str, base_dir: Path) -> Path:
    return base_dir / _sanitize_model_id(model_id) / "metadata.json"


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
    model_dir = _cache_dir_for_model(model_id, base)
    return model_dir / f"{quantization}.gguf"


def is_cached(
    model_id: str,
    quantization: str,
    cache_dir: Optional[str] = None,
) -> bool:
    """Check if a model is already cached with the given quantization.

    Args:
        model_id: HuggingFace model ID.
        quantization: Quantization level.
        cache_dir: Base cache directory.

    Returns:
        True if the cached GGUF file exists.
    """
    path = get_cache_path(model_id, quantization, cache_dir)
    return path.exists() and path.stat().st_size > 0


def list_cached(cache_dir: Optional[str] = None) -> list[CachedModel]:
    """List all cached models.

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

        model_id = model_dir.name.replace("--", "/")
        metadata_file = model_dir / "metadata.json"

        gguf_dir = model_dir / "gguf"
        if not gguf_dir.exists():
            continue

        for gguf_file in sorted(gguf_dir.glob("*.gguf")):
            quant = gguf_file.stem
            source = "unknown"

            # Try to read metadata
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        meta = json.load(f)
                    source = meta.get("source", "unknown")
                except (json.JSONDecodeError, OSError):
                    pass

            cached.append(CachedModel(
                model_id=model_id,
                gguf_path=str(gguf_file),
                quantization=quant,
                file_size=gguf_file.stat().st_size,
                source=source,
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
) -> int:
    """Delete a cached model file.

    Args:
        model_id: HuggingFace model ID.
        quantization: Quantization level.
        cache_dir: Base cache directory.

    Returns:
        Number of bytes freed.

    Raises:
        FileNotFoundError: If the model is not cached.
    """
    path = get_cache_path(model_id, quantization, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached model found for '{model_id}' at quantization '{quantization}'"
        )

    size = path.stat().st_size

    # Remove the GGUF file (or symlink)
    path.unlink()

    # Clean up empty directories
    gguf_dir = path.parent
    if gguf_dir.exists() and not any(gguf_dir.iterdir()):
        gguf_dir.rmdir()
        model_dir = gguf_dir.parent
        # Remove metadata too
        meta = model_dir / "metadata.json"
        if meta.exists():
            meta.unlink()
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()

    logger.info("Deleted cached model: %s (%s), freed %d bytes", model_id, quantization, size)
    return size


def download_model(
    model_id: str,
    quantization: str = "q4_k_m",
    cache_dir: Optional[str] = None,
    device: Optional["DeviceInfo"] = None,
    param_count_billions: float = 0.0,
) -> Path:
    """Download a model and return the path to the GGUF file.

    Strategy:
    1. Check if already cached.
    1b. Pre-download fit check (if device info available).
    2. Search for GGUF files in the original repo.
    3. Search for known GGUF variant repos (TheBloke, bartowski, etc.).
    4. If no GGUF found, raise a helpful error.

    Args:
        model_id: HuggingFace model ID.
        quantization: Desired quantization level.
        cache_dir: Base cache directory.
        device: Target device info for pre-download fit validation.
        param_count_billions: Estimated param count for fit check. 0 = skip.

    Returns:
        Path to the downloaded GGUF file.

    Raises:
        RuntimeError: If no GGUF file can be found for the model.
        ModelTooLargeError: If the model won't fit on the target device.
    """
    from huggingface_hub import hf_hub_download

    base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR

    # 1. Check cache
    cached_path = get_cache_path(model_id, quantization, cache_dir)
    if cached_path.exists() and cached_path.stat().st_size > 0:
        logger.info("Model already cached: %s", cached_path)
        return cached_path

    # 1b. Pre-download fit check — avoid downloading a model that won't fit
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

    # 2. Search for GGUF in the original repo
    gguf_filename = _search_gguf_in_repo(model_id, quantization)
    source_repo = model_id

    # 3. If not found, search for GGUF variant repos
    if gguf_filename is None:
        variant_repo = _find_gguf_variant_repo(model_id)
        if variant_repo:
            gguf_filename = _search_gguf_in_repo(variant_repo, quantization)
            if gguf_filename:
                source_repo = variant_repo

    # 4. If still no GGUF, raise error
    if gguf_filename is None:
        raise RuntimeError(
            f"No GGUF file found for model '{model_id}'.\n\n"
            f"UniInfer v0.1 requires pre-converted GGUF files. Options:\n"
            f"  1. Use a model that already has GGUF variants on HuggingFace\n"
            f"     (e.g., search for '{model_id.split('/')[-1]} GGUF' on huggingface.co)\n"
            f"  2. Provide a direct path to a local .gguf file\n"
            f"  3. Use a repo that already contains .gguf files\n"
        )

    # Download the GGUF file
    logger.info("Downloading %s from %s ...", gguf_filename, source_repo)

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
            # Try symlink first (saves disk space)
            cached_path.symlink_to(downloaded)
        except (OSError, NotImplementedError):
            # Fallback to copy on Windows or permission issues
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
        "source": "gguf_variant" if source_repo != model_id else "direct",
    }
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to write metadata: %s", exc)

    logger.info("Model cached at: %s", cached_path)
    return cached_path
