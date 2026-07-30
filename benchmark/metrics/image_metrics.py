from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np


def _float_image(image) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if value.size and value.max() > 1.0:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def mae(reference, prediction, mask=None) -> float:
    a, b = _float_image(reference), _float_image(prediction)
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} != {b.shape}")
    error = np.abs(a - b)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != a.shape[:2]:
            raise ValueError("Mask shape must match image height and width")
        error = error[selected]
    return float(error.mean()) if error.size else float("nan")


def psnr(reference, prediction, mask=None) -> float:
    a, b = _float_image(reference), _float_image(prediction)
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} != {b.shape}")
    squared = np.square(a - b)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != a.shape[:2]:
            raise ValueError("Mask shape must match image height and width")
        squared = squared[selected]
    error = float(squared.mean()) if squared.size else float("nan")
    return float("inf") if error == 0 else float(10.0 * math.log10(1.0 / error))


def ssim(reference, prediction, mask=None) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        warnings.warn("scikit-image unavailable; SSIM recorded as null")
        return None
    a, b = _float_image(reference), _float_image(prediction)
    score, full = structural_similarity(a, b, channel_axis=-1, data_range=1.0, full=True)
    if mask is None:
        return float(score)
    selected = np.asarray(mask, dtype=bool)
    per_pixel = full.mean(axis=-1) if full.ndim == 3 else full
    return float(per_pixel[selected].mean()) if selected.any() else None


@lru_cache(maxsize=1)
def _lpips_runtime():
    try:
        import lpips
        import torch
    except ImportError:
        return None
    model = lpips.LPIPS(net="alex")
    model.eval()
    return torch, model


def lpips_score(reference, prediction) -> float | None:
    runtime = _lpips_runtime()
    if runtime is None:
        return None
    torch, model = runtime
    tensors = []
    for image in (reference, prediction):
        value = _float_image(image)
        tensors.append(torch.from_numpy(value).permute(2, 0, 1)[None] * 2.0 - 1.0)
    with torch.no_grad():
        return float(model(*tensors).item())


def absolute_error_visualization(reference, prediction) -> np.ndarray:
    error = np.abs(_float_image(reference) - _float_image(prediction)).mean(axis=-1)
    error = np.clip(error * 4.0, 0.0, 1.0)
    return np.round(error * 255).astype(np.uint8)
