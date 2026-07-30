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
    "deep natural blue tones, a visible realistic star field with many small "
    "stars in the upper sky, subtle physically plausible stars, consistent "
    "horizon glow, preserve the exact skyline and cloud geometry, seamless 360 "
    "panorama, uniform exposure, continuous sky texture, no horizontal streaks, "
    "no bands, no aurora, no buildings, no landscape, no text"
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
    star_density: float = 0.00065
    star_luma_threshold: int = 145
    star_max_row_ratio: float = 0.58
    star_radius_px: int = 1
    sky_luma_cap: float = 0.42
    sky_hotspot_ratio: float = 1.55
    sky_hotspot_blur_fraction: float = 0.06
    sky_hotspot_strength: float = 0.85
    vae_tiling: bool = False
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
    if config.vae_tiling:
        pipe.enable_vae_tiling()
        print("[SkyRetexture] VAE tiling enabled; tiled decoding may introduce texture bands")
    elif hasattr(pipe, "disable_vae_tiling"):
        pipe.disable_vae_tiling()
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


def _compress_sky_hotspots(
    image: np.ndarray,
    sky_mask: np.ndarray,
    config: SkyRetextureConfig,
) -> tuple[np.ndarray, dict]:
    """Compress broad night-sky hotspots without flattening local texture."""
    out = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(sky_mask, dtype=bool)
    if not mask.any() or float(config.sky_luma_cap) <= 0:
        return out.copy(), {"adjusted_pixels": 0}

    rgb = out.astype(np.float32) / 255.0
    luma = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
    )
    h, w = mask.shape
    sigma_x = max(2.0, float(config.sky_hotspot_blur_fraction) * w)
    sigma_y = max(2.0, sigma_x * 0.35)
    pad = min(w // 4, max(8, int(round(3.0 * sigma_x))))
    mask_f = mask.astype(np.float32)
    weighted = luma * mask_f
    wrapped_weighted = np.concatenate(
        [weighted[:, -pad:], weighted, weighted[:, :pad]],
        axis=1,
    )
    wrapped_mask = np.concatenate(
        [mask_f[:, -pad:], mask_f, mask_f[:, :pad]],
        axis=1,
    )
    local_luma = cv2.GaussianBlur(
        wrapped_weighted,
        (0, 0),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
        borderType=cv2.BORDER_REFLECT,
    )[:, pad:pad + w]
    local_weight = cv2.GaussianBlur(
        wrapped_mask,
        (0, 0),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
        borderType=cv2.BORDER_REFLECT,
    )[:, pad:pad + w]
    local_luma /= np.maximum(local_weight, 1e-5)

    reference = float(np.median(local_luma[mask]))
    relative_cap = max(reference + 0.02, reference * float(config.sky_hotspot_ratio))
    cap = min(float(config.sky_luma_cap), relative_cap)
    affected = mask & (local_luma > cap)
    if not affected.any():
        return out.copy(), {
            "adjusted_pixels": 0,
            "reference_luma": reference,
            "effective_luma_cap": cap,
        }

    gain = np.ones_like(local_luma, dtype=np.float32)
    raw_gain = cap / np.maximum(local_luma, 1e-5)
    gain[affected] = np.power(
        np.clip(raw_gain[affected], 0.25, 1.0),
        float(config.sky_hotspot_strength),
    )
    corrected = rgb.copy()
    corrected[mask] *= gain[mask, None]
    corrected = np.clip(corrected * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return corrected, {
        "adjusted_pixels": int(affected.sum()),
        "reference_luma": reference,
        "effective_luma_cap": cap,
        "minimum_gain": float(gain[affected].min()),
    }


def _add_procedural_stars(
    image: np.ndarray,
    sky_mask: np.ndarray,
    config: SkyRetextureConfig,
) -> tuple[np.ndarray, int]:
    """Add a sparse star field to the upper dark sky while keeping the mask exact."""
    out = np.asarray(image, dtype=np.uint8).copy()
    sky_mask = np.asarray(sky_mask, dtype=bool)
    if not sky_mask.any():
        return out, 0

    h, w = out.shape[:2]
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    upper_limit = max(1, int(round(h * float(config.star_max_row_ratio))))
    candidate_mask = sky_mask.copy()
    candidate_mask[upper_limit:, :] = False
    candidate_mask &= gray <= int(config.star_luma_threshold)

    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return out, 0

    density = max(0.0, float(config.star_density))
    target = int(round(float(candidates.shape[0]) * density))
    target = max(0, min(target, 900))
    if target <= 0:
        return out, 0

    rng = np.random.default_rng(int(config.seed) ^ 0x6D2B79F5)
    # ERP rows oversample the poles. Weight by spherical area so the resulting
    # star field remains uniform when viewed as a sphere.
    spherical_weights = np.sin(
        np.pi * (candidates[:, 0].astype(np.float64) + 0.5) / float(h)
    )
    darkness = 1.0 - gray[candidates[:, 0], candidates[:, 1]].astype(np.float64) / 255.0
    weights = np.maximum(1e-8, spherical_weights * np.square(darkness))
    weights /= weights.sum()
    indices = rng.choice(
        candidates.shape[0],
        size=min(target, candidates.shape[0]),
        replace=False,
        p=weights,
    )
    star_points = candidates[indices]

    rgb = out.astype(np.float32) / 255.0
    radius_scale = max(0.5, float(config.star_radius_px))
    star_count = 0
    for y, x in star_points:
        if not sky_mask[y, x]:
            continue
        magnitude_sample = float(rng.random())
        peak = 0.10 + 0.78 * magnitude_sample**6
        sigma = radius_scale * (0.34 + 0.48 * magnitude_sample**3)
        radius = max(1, int(math.ceil(3.0 * sigma)))
        y0 = max(0, int(y) - radius)
        y1 = min(h, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(w, int(x) + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        subpixel_x = float(x) + float(rng.uniform(-0.45, 0.45))
        subpixel_y = float(y) + float(rng.uniform(-0.45, 0.45))
        psf = np.exp(
            -(
                np.square(xx - subpixel_x) + np.square(yy - subpixel_y)
            )
            / (2.0 * sigma * sigma)
        ).astype(np.float32)
        if rng.random() < 0.32:
            color = np.array([0.78, 0.88, 1.0], dtype=np.float32)
        elif rng.random() < 0.20:
            color = np.array([1.0, 0.90, 0.76], dtype=np.float32)
        else:
            color = np.array([0.96, 0.97, 1.0], dtype=np.float32)
        patch_mask = sky_mask[y0:y1, x0:x1]
        addition = psf[..., None] * peak * color
        rgb[y0:y1, x0:x1] = np.where(
            patch_mask[..., None],
            np.clip(rgb[y0:y1, x0:x1] + addition, 0.0, 1.0),
            rgb[y0:y1, x0:x1],
        )
        star_count += 1

    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8), star_count


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
    composite, hotspot_report = _compress_sky_hotspots(composite, mask, config)
    composite, star_count = _add_procedural_stars(composite, mask, config)
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
        "sky_hotspot_compression": hotspot_report,
        "procedural_star_count": int(star_count),
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
