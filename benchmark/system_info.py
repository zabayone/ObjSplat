from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

from benchmark.instrumentation.resources import memory_snapshot

PACKAGES = [
    "numpy", "Pillow", "plyfile", "psutil", "PyYAML", "matplotlib", "pandas",
    "scikit-image", "torch", "mlx", "mlx-gs", "transformers", "diffusers",
]
ENV_PREFIXES = (
    "OBJSPLAT_", "MLX_", "PYTORCH_", "TOKENIZERS_", "OMP_", "VECLIB_",
    "GROUNDING_", "SKY_", "MPS_", "METAL_", "CUDA_",
)


def _command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(f"System metadata command unavailable ({' '.join(args)}): {exc}")
        return None


def _sysctl(key: str) -> str | None:
    value = _command(["sysctl", "-n", key])
    return value.strip() if value else None


def _git(repo: Path, *args: str) -> str | None:
    return _command(["git", "-C", str(repo), *args])


def collect_system_info(
    repo_root: str | Path, *, argv: list[str] | None = None,
    scene_name: str | None = None, config_name: str | None = None,
    seed: int | None = None, panorama_path: str | Path | None = None,
) -> dict:
    repo_root = Path(repo_root).resolve()
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = memory_snapshot()
    commit = _git(repo_root, "rev-parse", "HEAD")
    dirty_text = _git(repo_root, "status", "--porcelain")
    pano = {"path": str(panorama_path) if panorama_path else None, "width": None, "height": None}
    if panorama_path and Path(panorama_path).exists():
        try:
            from PIL import Image
            with Image.open(panorama_path) as image:
                pano.update(width=image.width, height=image.height)
        except Exception as exc:
            warnings.warn(f"Could not read panorama dimensions: {exc}")
    chip = _sysctl("machdep.cpu.brand_string") if platform.system() == "Darwin" else platform.processor()
    machine_model = _sysctl("hw.model") if platform.system() == "Darwin" else platform.machine()
    unified = _sysctl("hw.memsize") if platform.system() == "Darwin" else None
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": commit, "dirty": bool(dirty_text), "status_porcelain": dirty_text or ""},
        "os": {
            "system": platform.system(), "release": platform.release(),
            "version": platform.version(), "platform": platform.platform(),
        },
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "hardware": {
            "machine_model": machine_model, "chip_model": chip,
            "architecture": platform.machine(),
            "total_unified_memory_bytes": int(unified) if unified and unified.isdigit() else memory.get("system_total_bytes"),
            "available_memory_before_bytes": memory.get("system_available_bytes"),
            "memory_semantics": (
                "Apple Silicon unified system/process memory estimates; not CUDA VRAM"
                if platform.system() == "Darwin" and platform.machine() == "arm64"
                else "system and process memory estimates"
            ),
        },
        "packages": packages,
        "device": {
            "requested": os.environ.get("DEVICE"),
            "mlx_available": packages.get("mlx") is not None,
            "torch_version": packages.get("torch"),
            "mlx_version": packages.get("mlx"),
        },
        "command_line": list(argv if argv is not None else sys.argv),
        "environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(ENV_PREFIXES)
        },
        "random_seeds": {"benchmark": seed},
        "panorama": pano,
        "scene_name": scene_name,
        "benchmark_configuration": config_name,
    }
