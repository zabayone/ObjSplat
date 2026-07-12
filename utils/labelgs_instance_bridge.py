#!/usr/bin/env python3
"""Bridge between optional instance-label maps and training point clouds.

This module provides a robust loader that projects 3D points into the
    equirectangular instance map produced by external tooling and returns
per-point instance ids.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def _find_path(save_dir: str, layer_idx: int) -> Optional[str]:
    candidates = [
        os.path.join(save_dir, "preprocess", "labelgs", "instances", f"layer{layer_idx}_instance_labels.npy"),
        os.path.join(save_dir, "instances", f"layer{layer_idx}_instance_labels.npy"),
        os.path.join(save_dir, f"layer{layer_idx}_instance_labels.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _find_3d_labels_path(save_dir: str, layer_idx: int) -> Optional[str]:
    candidates = [
        os.path.join(save_dir, "instances", f"layer{layer_idx}_labels_3d.npy"),
        os.path.join(save_dir, f"layer{layer_idx}_labels_3d.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_instance_labels_for_layer(
    save_dir: str,
    layer_idx: int,
    pcd_points: np.ndarray,
    pcd_masks: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    labels_3d_path = _find_3d_labels_path(save_dir, layer_idx)
    if labels_3d_path is not None:
        try:
            labels_3d = np.load(labels_3d_path).astype(np.int32).reshape(-1)
            if labels_3d.shape[0] == np.asarray(pcd_points).shape[0]:
                print(f"[INFO] Layer {layer_idx}: using compact 3D labels from {os.path.basename(labels_3d_path)}")
                return labels_3d
        except Exception as e:
            print(f"[WARNING] Failed to load 3D labels: {e}")

    instance_label_path = _find_path(save_dir, layer_idx)
    if instance_label_path is None:
        return None

    try:
        instance_map = np.load(instance_label_path).astype(np.int32)
        H, W = instance_map.shape
        pts = np.asarray(pcd_points, dtype=np.float32)

        # Equirectangular spherical projection: 3D -> (u,v)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        norm = np.sqrt(x * x + y * y + z * z)
        norm = np.where(norm < 1e-8, 1e-8, norm)

        lon = np.arctan2(x, z)          # [-pi, pi]
        lat = np.arcsin(np.clip(y / norm, -1.0, 1.0))  # [-pi/2, pi/2]

        u = ((lon / (2 * np.pi) + 0.5) * W).astype(np.int32)
        v = ((0.5 - lat / np.pi) * H).astype(np.int32)

        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        pcd_labels = instance_map[v, u].astype(np.int32)

        n_labeled = int((pcd_labels > 0).sum())
        print(
            f"[INFO] Layer {layer_idx}: {n_labeled}/{len(pcd_labels)} points labeled "
            f"({100 * n_labeled / max(len(pcd_labels),1):.1f}%) via equirect projection"
        )
        return pcd_labels

    except Exception as e:
        print(f"[WARNING] Failed to load instance labels: {e}")
        return None
