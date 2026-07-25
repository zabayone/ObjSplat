"""Masked day-to-night sky retexturing for equirectangular panoramas."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


DEFAULT_NIGHT_PROMPT = (
    "Transform only the masked daytime sky into a photorealistic clear night sky, "
    "deep natural blue tones, subtle physically plausible stars, consistent horizon "
    "glow, preserve the exact skyline and cloud geometry, seamless 360 panorama, "
    "uniform exposure, continuous sky texture, no horizontal streaks, no bands, "
    "no aurora, no buildings, no landscape, no text"
)


def _atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        image.save(tmp_path, format="PNG")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class SkyRetextureConfig:
    model_path: str
    prompt: str = DEFAULT_NIGHT_PROMPT
    seed: int = 42
    num_inference_steps: int = 50
    guidance_scale: float = 30.0
    max_pixels: int = 1024 * 1024
    mask_dilate_px: int = 5
    mask_feather_px: int = 9
    circular_padding_ratio: float = 0.0625
    seam_blend_px: int = 32
    min_sky_coverage: float = 0.005
    device: str = "mps"
    cpu_offload: bool = True
    validate_checkpoint: bool = True


def _round_multiple(value: float, multiple: int = 16) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _work_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    if max_pixels <= 0 or width * height <= max_pixels:
        return _round_multiple(width), _round_multiple(height)
    scale = math.sqrt(float(max_pixels) / float(width * height))
    return _round_multiple(width * scale), _round_multiple(height * scale)


def _load_inputs(scene_root: Path, metadata_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Layer metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sky = metadata.get("sky") or {}
    if sky.get("layer_idx") is None:
        # Backward compatibility for traindata generated before canonical sky
        # metadata was introduced.
        sky_group = next(
            (
                group
                for group in metadata.get("layer_groups", [])
                if str(group.get("group_label", "")).strip().lower() == "sky"
            ),
            None,
        )
        if sky_group is None:
            raise RuntimeError("Metadata does not contain a dedicated sky layer")
        layer_idx = int(sky_group["layer_idx"])
        sky = {
            "layer_idx": layer_idx,
            "instance_ids": [int(value) for value in sky_group.get("instance_ids", [])],
            "mask_path": f"traindata/layer{layer_idx}/layer{layer_idx}_erp_mask.png",
            "day_erp_path": f"traindata/layer{layer_idx}/layer{layer_idx}_erp_rgb.png",
            "night_erp_path": "traindata/sky/night_rgb.png",
            "role": "environment",
        }
        metadata["sky_layer_idx"] = layer_idx
        metadata["sky"] = sky

    source_path = scene_root / "rgb.png"
    if not source_path.exists() and metadata.get("input_dir"):
        source_path = Path(str(metadata["input_dir"])).expanduser() / "rgb.png"
    mask_path = scene_root / str(sky.get("mask_path", "traindata/sky/mask.png"))
    if not source_path.exists():
        raise FileNotFoundError(f"Source ERP not found: {source_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Sky mask not found: {mask_path}")

    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask.shape != source.shape[:2]:
        mask = cv2.resize(mask, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask >= 128
    if not mask.any():
        raise RuntimeError("Sky mask is empty")
    return source, mask, metadata


def _prepare_work_image(
    source: np.ndarray,
    mask: np.ndarray,
    config: SkyRetextureConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    src_h, src_w = source.shape[:2]
    work_w, work_h = _work_size(src_w, src_h, config.max_pixels)
    work_image = cv2.resize(source, (work_w, work_h), interpolation=cv2.INTER_LANCZOS4)
    work_mask = cv2.resize(mask.astype(np.uint8), (work_w, work_h), interpolation=cv2.INTER_NEAREST)

    if config.mask_dilate_px > 0:
        radius = max(1, int(round(config.mask_dilate_px * work_w / max(1, src_w))))
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        work_mask = cv2.dilate(work_mask, kernel, iterations=1)

    pad = _round_multiple(work_w * max(0.0, config.circular_padding_ratio))
    pad = min(pad, max(16, work_w // 4))
    padded_image = np.concatenate([work_image[:, -pad:], work_image, work_image[:, :pad]], axis=1)
    padded_mask = np.concatenate([work_mask[:, -pad:], work_mask, work_mask[:, :pad]], axis=1)
    return padded_image, padded_mask, pad


def _validate_flux_checkpoint(config: SkyRetextureConfig) -> Path:
    model_path = Path(config.model_path).expanduser().resolve()
    required = [
        model_path / "model_index.json",
        model_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        model_path / "vae" / "diffusion_pytorch_model.safetensors",
    ]
    index_path = model_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        required.extend(
            model_path / "transformer" / filename
            for filename in sorted(set(index.get("weight_map", {}).values()))
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete FLUX Fill checkpoint: " + ", ".join(missing))
    return model_path


def _load_pipeline(config: SkyRetextureConfig):
    import torch
    from diffusers import FluxFillPipeline

    model_path = _validate_flux_checkpoint(config)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.bfloat16 if config.device in {"mps", "cuda"} else torch.float32
    pipe = FluxFillPipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing("max")
        except Exception as exc:
            print(f"[SkyRetexture] Attention slicing unavailable: {exc}")

    if config.cpu_offload and config.device != "cpu":
        try:
            pipe.enable_model_cpu_offload(device=config.device)
        except Exception as exc:
            print(f"[SkyRetexture] CPU offload unavailable ({exc}); moving pipeline to {config.device}")
            pipe.to(config.device)
    else:
        pipe.to(config.device)
    return pipe


def _composite(
    source: np.ndarray,
    generated: np.ndarray,
    sky_mask: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    mask_f = sky_mask.astype(np.float32)
    if feather_px > 0:
        kernel = max(3, int(feather_px) * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        # Feather inward only: non-sky pixels always remain bit-identical.
        mask_f = cv2.GaussianBlur(mask_f, (kernel, kernel), 0) * mask_f
    alpha = mask_f[..., None]
    out = source.astype(np.float32) * (1.0 - alpha) + generated.astype(np.float32) * alpha
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _seam_error(image: np.ndarray, mask: np.ndarray) -> float:
    valid = mask[:, 0] & mask[:, -1]
    if not valid.any():
        return 0.0
    left = image[valid, 0].astype(np.float32)
    right = image[valid, -1].astype(np.float32)
    return float(np.mean(np.abs(left - right)))


def _harmonize_erp_seam(
    image: np.ndarray,
    mask: np.ndarray,
    band_px: int,
) -> np.ndarray:
    """Blend paired ERP edge pixels while leaving all non-sky pixels untouched."""
    out = np.asarray(image, dtype=np.float32).copy()
    width = out.shape[1]
    band = min(max(0, int(band_px)), max(0, width // 8))
    if band < 1:
        return np.asarray(image, dtype=np.uint8)
    source = out.copy()
    for offset in range(band):
        left_col = offset
        right_col = width - 1 - offset
        valid = mask[:, left_col] & mask[:, right_col]
        if not valid.any():
            continue
        strength = 1.0 - (float(offset) / float(max(1, band)))
        average = 0.5 * (
            source[valid, left_col] + source[valid, right_col]
        )
        out[valid, left_col] = (
            source[valid, left_col] * (1.0 - strength) + average * strength
        )
        out[valid, right_col] = (
            source[valid, right_col] * (1.0 - strength) + average * strength
        )
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def retexture_sky(
    scene_root: str | Path,
    config: SkyRetextureConfig,
    metadata_path: Optional[str | Path] = None,
    dry_run: bool = False,
) -> Path:
    """Generate a night ERP while preserving every non-sky source pixel."""
    scene_root = Path(scene_root).expanduser().resolve()
    metadata_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path
        else scene_root / "traindata" / "layer_instances.json"
    )
    source, mask, metadata = _load_inputs(scene_root, metadata_path)
    coverage = float(mask.mean())
    if coverage < float(config.min_sky_coverage):
        raise RuntimeError(
            f"Sky mask coverage is only {coverage:.4f}, below the safety threshold "
            f"{config.min_sky_coverage:.4f}. Rerun segmentation with the updated "
            "Grounding-SAM sky handling, or explicitly lower --min_sky_coverage."
        )
    output_dir = scene_root / "traindata" / "sky"
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.validate_checkpoint:
        _validate_flux_checkpoint(config)

    padded_image, padded_mask, pad = _prepare_work_image(source, mask, config)
    work_h, padded_w = padded_image.shape[:2]
    work_w = padded_w - 2 * pad
    print(
        f"[SkyRetexture] source={source.shape[1]}x{source.shape[0]} "
        f"work={work_w}x{work_h} padded={padded_w}x{work_h} "
        f"sky_coverage={coverage:.4f}"
    )
    if dry_run:
        return output_dir / "night_rgb.png"

    import torch

    pipe = _load_pipeline(config)
    generator = torch.Generator("cpu").manual_seed(int(config.seed))
    result = pipe(
        prompt=config.prompt,
        image=Image.fromarray(padded_image),
        mask_image=Image.fromarray(padded_mask.astype(np.uint8) * 255),
        height=work_h,
        width=padded_w,
        num_inference_steps=int(config.num_inference_steps),
        guidance_scale=float(config.guidance_scale),
        generator=generator,
        max_sequence_length=512,
    ).images[0]

    generated_padded = np.asarray(result.convert("RGB"), dtype=np.uint8)
    generated_work = generated_padded[:, pad : pad + work_w]
    generated = cv2.resize(
        generated_work,
        (source.shape[1], source.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    composite_raw = _composite(source, generated, mask, config.mask_feather_px)
    seam_before_harmonization = _seam_error(composite_raw, mask)
    composite = _harmonize_erp_seam(
        composite_raw,
        mask,
        band_px=config.seam_blend_px,
    )
    night_layer = np.zeros_like(composite)
    night_layer[mask] = composite[mask]

    composite_path = output_dir / "night_composite.png"
    layer_path = output_dir / "night_rgb.png"
    _atomic_save_png(Image.fromarray(composite), composite_path)
    _atomic_save_png(Image.fromarray(night_layer), layer_path)

    generation = {
        "status": "complete",
        "config": asdict(config),
        "source_size": [int(source.shape[1]), int(source.shape[0])],
        "work_size": [int(work_w), int(work_h)],
        "padded_work_size": [int(padded_w), int(work_h)],
        "sky_coverage": coverage,
        "seam_mae_before": _seam_error(source, mask),
        "seam_mae_generated": seam_before_harmonization,
        "seam_mae_after": _seam_error(composite, mask),
        "night_layer_path": str(layer_path.relative_to(scene_root)),
        "night_composite_path": str(composite_path.relative_to(scene_root)),
    }
    _atomic_write_json(generation, output_dir / "night_generation.json")

    sky_meta = metadata.setdefault("sky", {})
    sky_meta["night_erp_path"] = str(layer_path.relative_to(scene_root))
    sky_meta["night_composite_path"] = str(composite_path.relative_to(scene_root))
    sky_meta["night_generation_path"] = str(
        (output_dir / "night_generation.json").relative_to(scene_root)
    )
    _atomic_write_json(metadata, metadata_path)
    print(f"[SkyRetexture] Night sky written to {layer_path}")
    print(f"[SkyRetexture] Full night ERP written to {composite_path}")
    return layer_path
