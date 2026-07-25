"""Build efficient day/night ERP and Gaussian parameter variants."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData


SH_C0 = 0.28209479177387814


@dataclass
class NightMoodConfig:
    exposure_ev: float = -2.65
    contrast: float = 0.98
    saturation: float = 0.32
    blue_tint_r: float = 0.44
    blue_tint_g: float = 0.64
    blue_tint_b: float = 1.12
    ambient_floor: float = 0.012
    highlight_preservation: float = 0.08
    shadow_suppression: float = 0.82
    illumination_flattening: float = 0.55
    shadow_blur_fraction: float = 0.035
    shadow_threshold: float = 0.82
    shadow_max_lift: float = 2.4
    gaussian_color_strength: float = 1.0
    gaussian_sky_strength: float = 1.0
    directional_sh_scale: float = 0.55
    sky_directional_sh_scale: float = 0.20
    chunk_rows: int = 256
    gaussian_chunk_size: int = 1_000_000


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        image.save(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _estimate_shadow_lift(
    rgb_u8: np.ndarray,
    config: NightMoodConfig,
    exclusion_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a low-frequency multiplier that suppresses daytime cast shadows."""
    height, width = rgb_u8.shape[:2]
    analysis_max_side = 1024
    scale = min(1.0, analysis_max_side / float(max(height, width)))
    analysis_size = (
        max(32, int(round(width * scale))),
        max(16, int(round(height * scale))),
    )
    small = cv2.resize(rgb_u8, analysis_size, interpolation=cv2.INTER_AREA)
    small_linear = _srgb_to_linear(small.astype(np.float32) / 255.0)
    luma = np.sum(
        small_linear * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
    )

    # Blur in log space with horizontal wrapping so the ERP seam does not
    # receive a different illumination estimate on its two sides.
    sigma = max(1.0, float(config.shadow_blur_fraction) * analysis_size[0])
    pad = min(analysis_size[0] // 4, max(8, int(round(3.0 * sigma))))
    log_luma = np.log(np.maximum(luma, 1e-4))
    wrapped = np.concatenate([log_luma[:, -pad:], log_luma, log_luma[:, :pad]], axis=1)
    illumination = cv2.GaussianBlur(
        wrapped,
        (0, 0),
        sigmaX=sigma,
        sigmaY=max(1.0, sigma * 0.55),
        borderType=cv2.BORDER_REFLECT,
    )[:, pad:pad + analysis_size[0]]
    illumination = np.exp(illumination)

    ratio = luma / np.maximum(illumination, 1e-4)
    shadow = 1.0 - _smoothstep(
        max(0.15, float(config.shadow_threshold) * 0.45),
        float(config.shadow_threshold),
        ratio,
    )
    desired = np.minimum(
        float(config.shadow_max_lift),
        np.maximum(1.0, illumination * 0.82 / np.maximum(luma, 1e-4)),
    )
    multiplier = 1.0 + float(config.shadow_suppression) * shadow * (desired - 1.0)

    # Daylight often has a broad directional gradient that is too slow-varying
    # to be classified as a local shadow. Flatten that illumination field
    # toward the median non-sky level before applying the night exposure.
    valid = np.ones_like(illumination, dtype=bool)
    if exclusion_mask is not None:
        small_exclusion = cv2.resize(
            np.asarray(exclusion_mask, dtype=np.uint8),
            analysis_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid &= ~small_exclusion
    reference = float(np.median(illumination[valid])) if valid.any() else float(
        np.median(illumination)
    )
    flatten_gain = np.power(
        reference / np.maximum(illumination, 1e-4),
        float(config.illumination_flattening),
    )
    multiplier *= np.clip(flatten_gain, 0.55, 1.85)
    multiplier = cv2.resize(
        multiplier.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    np.clip(
        multiplier,
        0.55,
        float(config.shadow_max_lift),
        out=multiplier,
    )
    return multiplier.astype(np.float16)


def _relight_non_sky(
    rgb_u8: np.ndarray,
    config: NightMoodConfig,
    shadow_lift: Optional[np.ndarray] = None,
) -> np.ndarray:
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    linear = _srgb_to_linear(rgb)
    source_luma = np.sum(
        linear * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    if shadow_lift is not None:
        linear *= np.asarray(shadow_lift, dtype=np.float32)[..., None]

    relit = linear * float(2.0 ** config.exposure_ev)
    relit = 0.18 * np.power(
        np.maximum(relit, 0.0) / 0.18,
        float(config.contrast),
    )
    relit_luma = np.sum(
        relit * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    relit = relit_luma + (relit - relit_luma) * float(config.saturation)
    relit *= np.array(
        [config.blue_tint_r, config.blue_tint_g, config.blue_tint_b],
        dtype=np.float32,
    )
    relit += float(config.ambient_floor) * np.array(
        [0.28, 0.46, 1.0], dtype=np.float32
    )

    # Preserve only the strongest emissive-looking highlights. Broad sunlit
    # areas must be crushed with the rest of the scene at night.
    highlight = _smoothstep(0.58, 0.94, source_luma)
    preserved = linear * 0.58
    mix = highlight * float(config.highlight_preservation)
    relit = relit * (1.0 - mix) + preserved * mix
    return np.clip(_linear_to_srgb(relit) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def build_night_scene_erp(
    scene_root: str | Path,
    config: NightMoodConfig,
    metadata_path: Optional[str | Path] = None,
) -> Path:
    """Relight non-sky pixels and combine them with the generated night sky."""
    scene_root = Path(scene_root).expanduser().resolve()
    metadata_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path
        else scene_root / "traindata" / "layer_instances.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sky_meta = metadata.get("sky") or {}
    source_path = scene_root / "rgb.png"
    mask_path = scene_root / str(sky_meta.get("mask_path", "traindata/sky/mask.png"))
    night_sky_path = scene_root / str(
        sky_meta.get("night_composite_path", "traindata/sky/night_composite.png")
    )
    for path in (source_path, mask_path, night_sky_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing mood input: {path}")

    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
    night_sky = np.asarray(Image.open(night_sky_path).convert("RGB"), dtype=np.uint8)
    sky_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 128
    if source.shape != night_sky.shape or sky_mask.shape != source.shape[:2]:
        raise ValueError(
            f"Mood input shape mismatch: source={source.shape}, "
            f"night={night_sky.shape}, mask={sky_mask.shape}"
        )

    output = np.empty_like(source)
    shadow_lift = _estimate_shadow_lift(source, config, exclusion_mask=sky_mask)
    chunk_rows = max(1, int(config.chunk_rows))
    for row0 in range(0, source.shape[0], chunk_rows):
        row1 = min(source.shape[0], row0 + chunk_rows)
        relit = _relight_non_sky(
            source[row0:row1],
            config,
            shadow_lift=shadow_lift[row0:row1],
        )
        mask_chunk = sky_mask[row0:row1]
        output[row0:row1] = np.where(
            mask_chunk[..., None],
            night_sky[row0:row1],
            relit,
        )

    mood_dir = scene_root / "traindata" / "moods" / "night"
    scene_path = mood_dir / "scene_rgb.png"
    _atomic_save_image(Image.fromarray(output), scene_path)
    manifest = {
        "name": "night",
        "status": "erp_ready",
        "scene_erp_path": str(scene_path.relative_to(scene_root)),
        "source_day_erp_path": str(source_path.relative_to(scene_root)),
        "sky_mask_path": str(mask_path.relative_to(scene_root)),
        "night_sky_composite_path": str(night_sky_path.relative_to(scene_root)),
        "config": asdict(config),
    }
    _atomic_write_json(manifest, mood_dir / "mood.json")
    metadata.setdefault("moods", {})["night"] = manifest
    _atomic_write_json(metadata, metadata_path)
    return scene_path


def _erp_indices_for_points(
    xyz: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    theta = np.arctan2(x, z)
    horizontal = np.sqrt(x * x + z * z)
    phi = np.arctan2(-y, horizontal)
    u = np.floor((theta + np.pi) * (width / (2.0 * np.pi))).astype(np.int64)
    v = np.floor((0.5 * np.pi - phi) * (height / np.pi)).astype(np.int64)
    return np.mod(u, width), np.clip(v, 0, height - 1)


def adapt_gaussian_ply_to_erp(
    source_ply: str | Path,
    target_ply: str | Path,
    target_erp_path: str | Path,
    sky_mask_path: str | Path,
    config: NightMoodConfig,
    sky_radius: float = 0.0,
) -> dict:
    """Fit SH colors to an ERP while preserving every Gaussian and its geometry."""
    source_ply = Path(source_ply).expanduser().resolve()
    target_ply = Path(target_ply).expanduser().resolve()
    target_erp = np.asarray(
        Image.open(target_erp_path).convert("RGB"), dtype=np.uint8
    )
    sky_mask = np.asarray(
        Image.open(sky_mask_path).convert("L"), dtype=np.uint8
    ) >= 128
    if sky_mask.shape != target_erp.shape[:2]:
        raise ValueError("Sky mask and target ERP have different shapes")

    source_header = PlyData.read(source_ply, mmap="r")
    if source_header.text:
        raise ValueError(
            "Efficient mood adaptation requires a binary PLY; convert the ASCII PLY first"
        )
    source_names = source_header["vertex"].data.dtype.names or ()
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required.difference(source_names))
    if missing:
        raise ValueError(f"PLY is missing Gaussian fields: {missing}")
    target_ply.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_ply.stem}.", suffix=".ply", dir=target_ply.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    shutil.copyfile(source_ply, tmp_path)
    target_stream = open(tmp_path, "r+b")
    ply = PlyData.read(target_stream, mmap="r+")
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()

    rest_fields = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    chunk_size = max(1, int(config.gaussian_chunk_size))
    sky_count = 0
    color_delta_sum = 0.0
    for start in range(0, len(vertex), chunk_size):
        end = min(len(vertex), start + chunk_size)
        xyz = np.stack(
            [vertex["x"][start:end], vertex["y"][start:end], vertex["z"][start:end]],
            axis=1,
        ).astype(np.float32)
        u, v = _erp_indices_for_points(
            xyz, target_erp.shape[1], target_erp.shape[0]
        )
        target_rgb = target_erp[v, u].astype(np.float32) / 255.0
        original_rgb = np.stack(
            [
                vertex["f_dc_0"][start:end],
                vertex["f_dc_1"][start:end],
                vertex["f_dc_2"][start:end],
            ],
            axis=1,
        ).astype(np.float32) * SH_C0 + 0.5
        radius = np.linalg.norm(xyz, axis=1)
        is_sky = sky_mask[v, u]
        if float(sky_radius) > 0:
            is_sky &= radius >= float(sky_radius) * 0.80
        strength = np.where(
            is_sky,
            float(config.gaussian_sky_strength),
            float(config.gaussian_color_strength),
        )[:, None]
        fitted_rgb = np.clip(
            original_rgb * (1.0 - strength) + target_rgb * strength,
            0.0,
            1.0,
        )
        fitted_dc = (fitted_rgb - 0.5) / SH_C0
        for channel, field in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
            vertex[field][start:end] = fitted_dc[:, channel]

        if rest_fields:
            directional_scale = np.where(
                is_sky,
                float(config.sky_directional_sh_scale),
                float(config.directional_sh_scale),
            ).astype(np.float32)
            for field in rest_fields:
                vertex[field][start:end] *= directional_scale
        sky_count += int(is_sky.sum())
        color_delta_sum += float(np.abs(fitted_rgb - original_rgb).sum())

    try:
        vertex.flush()
        target_stream.flush()
        os.fsync(target_stream.fileno())
        target_stream.close()
        os.replace(tmp_path, target_ply)
    finally:
        if not target_stream.closed:
            target_stream.close()
        tmp_path.unlink(missing_ok=True)

    return {
        "source_ply": str(source_ply),
        "target_ply": str(target_ply),
        "gaussian_count": int(len(vertex)),
        "sky_gaussian_count": int(sky_count),
        "mean_absolute_rgb_delta": (
            color_delta_sum / max(1, len(vertex) * 3)
        ),
        "geometry_preserved": True,
        "opacity_preserved": True,
    }


def write_mood_manifest(scene_root: str | Path, payload: dict) -> Path:
    scene_root = Path(scene_root).expanduser().resolve()
    manifest_path = scene_root / "scene" / "moods.json"
    current = {}
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
    current.setdefault("version", 1)
    current.setdefault("moods", {}).update(payload.get("moods", {}))
    if payload.get("active_mood"):
        current["active_mood"] = payload["active_mood"]
    _atomic_write_json(current, manifest_path)
    return manifest_path
