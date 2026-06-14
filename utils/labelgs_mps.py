from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from plyfile import PlyData, PlyElement


def infer_point_labels(traindata: Dict[str, Any]) -> np.ndarray:
    explicit_labels = traindata.get("pcd_labels")
    if explicit_labels is not None:
        arr = np.asarray(explicit_labels)
        return np.round(arr).astype(np.int32).reshape(-1)

    # Prova il bridge su disco
    save_dir  = traindata.get("save_dir")
    layer_idx = traindata.get("layer_idx")
    pcd_points = np.asarray(traindata["pcd_points"])
    if save_dir is not None and layer_idx is not None:
        try:
            from utils.labelgs_instance_bridge import load_instance_labels_for_layer
            labels = load_instance_labels_for_layer(save_dir, layer_idx, pcd_points)
            if labels is not None:
                return labels.astype(np.int32)
        except Exception as e:
            print(f"[WARNING] labelgs_instance_bridge fallito: {e}")

    # Fallback maschere
    masks = traindata.get("pcd_masks")
    if masks is None:
        return np.zeros((pcd_points.shape[0],), dtype=np.int32)

    mask_array = np.asarray(masks, dtype=np.float32)
    if mask_array.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if mask_array.ndim == 1:
        signal = mask_array
    elif mask_array.ndim == 2:
        signal = mask_array.mean(axis=1)
    else:
        signal = mask_array.reshape(mask_array.shape[0], -1).mean(axis=1)
    if signal.max() > 1.0:
        signal = signal / signal.max()
    return (signal > 0.5).astype(np.int32)


def write_layerpano_compatible_ply(
    path: str,
    xyz: np.ndarray,
    normals: np.ndarray,
    f_dc: np.ndarray,
    f_rest: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> None:
    N = xyz.shape[0]
    dtype_list = []
    arrays = {}

    rots_np = np.asarray(rots, dtype=np.float32)
    if rots_np.ndim == 2 and rots_np.shape[1] == 4:
        norms = np.linalg.norm(rots_np, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        rots_np = rots_np / norms

    for i, name in enumerate(["x", "y", "z"]):
        dtype_list.append((name, np.float32)); arrays[name] = xyz[:, i]
    for i, name in enumerate(["nx", "ny", "nz"]):
        dtype_list.append((name, np.float32)); arrays[name] = normals[:, i]
    for i in range(3):
        dtype_list.append((f"f_dc_{i}", np.float32)); arrays[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(f_rest.shape[1]):
        dtype_list.append((f"f_rest_{i}", np.float32)); arrays[f"f_rest_{i}"] = f_rest[:, i]
    dtype_list.append(("opacity", np.float32))
    arrays["opacity"] = opacities.reshape(-1)
    for i in range(3):
        dtype_list.append((f"scale_{i}", np.float32)); arrays[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        dtype_list.append((f"rot_{i}", np.float32)); arrays[f"rot_{i}"] = rots_np[:, i]
    if labels is not None:
        dtype_list.append(("label", np.int32))
        arrays["label"] = np.asarray(labels, dtype=np.int32).reshape(-1)

    vertex_array = np.zeros(N, dtype=dtype_list)
    for name, _ in dtype_list:
        vertex_array[name] = arrays[name]

    el = PlyElement.describe(vertex_array, "vertex")
    PlyData([el]).write(path)