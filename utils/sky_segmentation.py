"""Lightweight semantic sky segmentation for ERP panoramas."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _resize_preserving_aspect(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _sky_label_id(id2label: dict[Any, str]) -> int:
    for key, value in id2label.items():
        label = str(value).strip().lower().replace("_", " ")
        if label == "sky":
            return int(key)
    raise RuntimeError("The semantic segmentation model has no 'sky' class")


def _clean_sky_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    h, w = mask_u8.shape
    kernel_size = max(3, int(round(w / 800.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_u8, dtype=bool)
    top_band = max(1, int(round(h * 0.06)))
    min_area = max(64, int(round(h * w * 0.00025)))
    for idx in range(1, n_labels):
        component = labels == idx
        area = int(stats[idx, cv2.CC_STAT_AREA])
        centroid_y = float(centroids[idx][1])
        touches_zenith = bool(component[:top_band].any())
        if touches_zenith or (area >= min_area and centroid_y < h * 0.55):
            keep |= component

    # The lower quarter surrounds the nadir and cannot be outdoor sky in a
    # correctly oriented ERP. This also suppresses blue pavement/water errors.
    keep[int(round(h * 0.78)) :, :] = False
    return keep


def _protect_thin_dark_details(image: np.ndarray, sky_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove wire/branch-like dark details from an otherwise semantic sky mask."""
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    width = gray.shape[1]
    kernel_size = max(5, int(round(width / 900.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    details = (blackhat >= 24) & sky_mask
    if details.any():
        details = cv2.dilate(
            details.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    protected = np.asarray(sky_mask, dtype=bool) & ~details
    return protected, int(details.sum())


def _fill_zenith_cap(sky_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill a small zenith cap when the surrounding upper sky is already present."""
    mask = np.asarray(sky_mask, dtype=bool).copy()
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return mask, 0

    cap_h = max(4, int(round(h * 0.16)))
    cap_w = max(8, int(round(w * 0.22)))
    yy, xx = np.ogrid[:cap_h, :w]
    cx = (w - 1) * 0.5
    ellipse = ((xx - cx) / max(1.0, cap_w * 0.5)) ** 2 + (yy / max(1.0, cap_h)) ** 2 <= 1.0

    center_x1 = max(0, int(round(w * 0.36)))
    center_x2 = min(w, int(round(w * 0.64)))
    if center_x2 <= center_x1:
        return mask, 0

    upper_band = mask[:cap_h, :]
    center_band = mask[:cap_h, center_x1:center_x2]
    support_band = mask[: max(1, int(round(cap_h * 0.75))), :]

    # Only fill if the top sky is already established around the hole. This
    # keeps the heuristic conservative for scenes with genuine overhead cover.
    support_coverage = float(support_band.mean()) if support_band.size else 0.0
    center_coverage = float(center_band.mean()) if center_band.size else 0.0
    if support_coverage < 0.05 or center_coverage >= 0.20:
        return mask, 0

    fill = ellipse & ~mask[:cap_h, :]
    if not fill.any():
        return mask, 0
    top = mask[:cap_h, :].copy()
    top[fill] = True
    mask[:cap_h, :] = top
    return mask, int(fill.sum())


def segment_sky_segformer(
    pano_rgb: np.ndarray,
    model_id: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    device: str = "mps",
    max_side: int = 2048,
    threshold: float = 0.45,
    seam_ensemble: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return an ERP sky mask and diagnostic data.

    Two predictions with the panorama shifted by half its width reduce the
    influence of the artificial left/right ERP boundary.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    source = np.asarray(pano_rgb, dtype=np.uint8)
    work = _resize_preserving_aspect(source, max(256, int(max_side)))
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    sky_id = _sky_label_id(model.config.id2label)
    target_device = torch.device(device)
    model = model.to(target_device).eval()

    def predict(image: np.ndarray) -> np.ndarray:
        inputs = processor(images=Image.fromarray(image), return_tensors="pt", do_resize=False)
        pixel_values = inputs["pixel_values"].to(target_device)
        with torch.inference_mode():
            logits = model(pixel_values=pixel_values).logits
            logits = F.interpolate(
                logits,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            probability = torch.softmax(logits.float(), dim=1)[0, sky_id]
        return probability.detach().cpu().numpy().astype(np.float32)

    probability = predict(work)
    if seam_ensemble:
        shift = work.shape[1] // 2
        rolled = np.roll(work, shift=shift, axis=1)
        rolled_probability = predict(rolled)
        probability = 0.5 * (probability + np.roll(rolled_probability, shift=-shift, axis=1))

    raw_mask = probability >= float(threshold)
    clean_mask = _clean_sky_mask(raw_mask)
    full_mask = cv2.resize(
        clean_mask.astype(np.uint8),
        (source.shape[1], source.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    full_mask, protected_dark_pixels = _protect_thin_dark_details(source, full_mask)
    full_mask, zenith_cap_pixels = _fill_zenith_cap(full_mask)
    diagnostics = {
        "model": model_id,
        "work_size": [int(work.shape[1]), int(work.shape[0])],
        "threshold": float(threshold),
        "seam_ensemble": bool(seam_ensemble),
        "raw_coverage": float(raw_mask.mean()),
        "coverage": float(full_mask.mean()),
        "mean_sky_probability": float(probability[clean_mask].mean()) if clean_mask.any() else 0.0,
        "protected_dark_detail_pixels": int(protected_dark_pixels),
        "zenith_cap_pixels": int(zenith_cap_pixels),
    }
    del model
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return full_mask, diagnostics
