from __future__ import annotations

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

from benchmark.io_utils import atomic_json


@lru_cache(maxsize=32)
def _sha256_cached(path_value: str, size_bytes: int, mtime_ns: int) -> str:
    path = Path(path_value)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _sha256_cached(str(path), stat.st_size, stat.st_mtime_ns)


def prepare_panorama(
    source: str | Path,
    scene_root: str | Path,
    *,
    target_width: int | None = None,
    require_equirectangular_2_to_1: bool = True,
) -> dict:
    """Convert a JPEG/PNG ERP into a reproducible RGB PNG working input."""
    source = Path(source).expanduser().resolve()
    scene_root = Path(scene_root).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input panorama does not exist: {source}")
    scene_root.mkdir(parents=True, exist_ok=True)
    target = scene_root / "rgb.png"
    manifest_path = scene_root / "input_preparation.json"
    source_hash = file_sha256(source)

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source) as handle:
        source_format = handle.format
        source_mode = handle.mode
        image = ImageOps.exif_transpose(handle).convert("RGB")
        source_width, source_height = image.size
        ratio = source_width / source_height
        if require_equirectangular_2_to_1 and abs(ratio - 2.0) > 1e-3:
            raise ValueError(
                f"{source} is {source_width}x{source_height} ({ratio:.6f}:1), "
                "but an equirectangular 2:1 panorama is required"
            )
        if target_width is not None:
            target_width = int(target_width)
            if target_width < 512 or target_width % 2:
                raise ValueError("input_preprocessing.target_width must be an even integer >= 512")
            target_size = (target_width, target_width // 2)
        else:
            target_size = image.size

        manifest = {
            "source_path": str(source),
            "source_sha256": source_hash,
            "source_format": source_format,
            "source_mode": source_mode,
            "source_width": source_width,
            "source_height": source_height,
            "source_aspect_ratio": ratio,
            "target_path": str(target),
            "target_format": "PNG",
            "target_mode": "RGB",
            "target_width": target_size[0],
            "target_height": target_size[1],
            "resize_filter": "Lanczos" if target_size != image.size else None,
            "aspect_policy": "require_2_to_1",
        }
        if target.exists() and manifest_path.exists():
            import json
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            comparable = (
                existing.get("source_sha256") == source_hash
                and existing.get("target_width") == target_size[0]
                and existing.get("target_height") == target_size[1]
            )
            if comparable:
                return existing
            raise RuntimeError(
                f"{scene_root} already contains a working panorama from a different "
                "source or preprocessing configuration. Use a new scene_root to "
                "avoid mixing stale depth/training artifacts."
            )
        if target.exists() != manifest_path.exists():
            raise RuntimeError(
                f"Incomplete existing input state in {scene_root}; use a new scene_root "
                "or remove the stale working directory deliberately."
            )
        if target_size != image.size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        fd, tmp_name = tempfile.mkstemp(prefix=".rgb.", suffix=".png", dir=scene_root)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            image.save(tmp_path, format="PNG", compress_level=3)
            os.replace(tmp_path, target)
        finally:
            tmp_path.unlink(missing_ok=True)
        atomic_json(manifest_path, manifest)
        return manifest
