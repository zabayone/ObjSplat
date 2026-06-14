#!/usr/bin/env python3
"""Propagate instance labels from equirectangular instance maps to pointclouds.

Strategy:
- Project 3D points to instance_map (u,v) and read labels
- For unlabeled points, perform local window majority (dilation-like)
- For still-unlabeled, assign nearest neighbor label using KDTree within threshold

Writes: propagated labels numpy and optional PLY with updated `label` field.
"""
from __future__ import annotations
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
import os
from scipy.spatial import cKDTree


def project_points_to_uv(pts: np.ndarray, H: int, W: int):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    norm = np.sqrt(x * x + y * y + z * z)
    norm = np.where(norm < 1e-8, 1e-8, norm)
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y / norm, -1.0, 1.0))
    u = ((lon / (2 * np.pi) + 0.5) * W).astype(np.int32)
    v = ((0.5 - lat / np.pi) * H).astype(np.int32)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    return u, v


def load_instance_map(save_dir: str, layer_idx: int):
    candidates = [
        os.path.join(save_dir, "preprocess", "labelgs", "instances", f"layer{layer_idx}_instance_labels.npy"),
        os.path.join(save_dir, "instances", f"layer{layer_idx}_instance_labels.npy"),
        os.path.join(save_dir, f"layer{layer_idx}_instance_labels.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return np.load(p).astype(np.int32)
    return None


def local_window_majority(instance_map: np.ndarray, u: np.ndarray, v: np.ndarray, radii=(1, 3, 5)):
    H, W = instance_map.shape
    N = u.shape[0]
    result = np.zeros(N, dtype=np.int32)
    unlabeled_mask = np.ones(N, dtype=bool)
    for r in radii:
        if not unlabeled_mask.any():
            break
        dx = np.arange(-r, r + 1, 1)
        dy = np.arange(-r, r + 1, 1)
        labels_stack = []
        uu = u[unlabeled_mask]
        vv = v[unlabeled_mask]
        for ddx in dx:
            for ddy in dy:
                cu = np.clip(uu + ddx, 0, W - 1)
                cv = np.clip(vv + ddy, 0, H - 1)
                labels_stack.append(instance_map[cv, cu])
        labels_stack = np.stack(labels_stack, axis=1)  # (M, K)
        # zero means background; ignore zeros
        import scipy.stats as stats
        mode_labels = np.zeros(labels_stack.shape[0], dtype=np.int32)
        for i in range(labels_stack.shape[0]):
            vals = labels_stack[i, labels_stack[i] != 0]
            if vals.size == 0:
                mode_labels[i] = 0
            else:
                mode_labels[i] = stats.mode(vals, keepdims=False).mode
        assign_idx = np.where(mode_labels != 0)[0]
        global_idx = np.nonzero(unlabeled_mask)[0][assign_idx]
        result[global_idx] = mode_labels[assign_idx]
        unlabeled_mask[global_idx] = False
    return result, unlabeled_mask


def write_ply_with_labels(in_ply: str, out_ply: str, new_labels: np.ndarray):
    ply = PlyData.read(in_ply)
    v = ply['vertex']
    N = v.count
    dtype_names = v.data.dtype.names
    # prepare arrays
    arrays = {name: np.asarray(v[name]) for name in dtype_names}
    arrays['label'] = np.asarray(new_labels, dtype=np.int32)
    # build dtype list
    dtype_list = []
    for name in ['x', 'y', 'z']:
        dtype_list.append((name, np.float32))
    for name in ['red', 'green', 'blue']:
        dtype_list.append((name, np.uint8))
    dtype_list.append(('label', np.int32))
    vertex_array = np.zeros(N, dtype=dtype_list)
    vertex_array['x'] = arrays['x']
    vertex_array['y'] = arrays['y']
    vertex_array['z'] = arrays['z']
    vertex_array['red'] = arrays.get('red', np.zeros(N, dtype=np.uint8))
    vertex_array['green'] = arrays.get('green', np.zeros(N, dtype=np.uint8))
    vertex_array['blue'] = arrays.get('blue', np.zeros(N, dtype=np.uint8))
    vertex_array['label'] = arrays['label']
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el]).write(out_ply)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--ply', required=True)
    p.add_argument('--out_ply', required=False)
    p.add_argument('--out_np', required=False)
    p.add_argument('--knn_radius', type=float, default=0.02)
    p.add_argument('--max_knn', type=int, default=1)
    args = p.parse_args()

    instance_map = load_instance_map(args.save_dir, args.layer)
    if instance_map is None:
        print('No instance_map found for layer', args.layer)
        return
    H, W = instance_map.shape
    print('Loaded instance_map', instance_map.shape, 'unique', np.unique(instance_map)[:10])

    ply = PlyData.read(args.ply)
    v = ply['vertex']
    pts = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    N = pts.shape[0]
    print('Loaded PLY points', N)

    u, vuv = project_points_to_uv(pts, H, W)
    labels = instance_map[vuv, u].astype(np.int32)
    n_labeled = int((labels > 0).sum())
    print(f'Initially labeled {n_labeled}/{N} ({100*n_labeled/N:.3f}%)')

    # local window majority
    propagated, still_unlabeled_mask = local_window_majority(instance_map, u, vuv, radii=(1, 3))
    labels[~still_unlabeled_mask & (labels == 0)] = propagated[~still_unlabeled_mask & (labels == 0)]
    n_labeled2 = int((labels > 0).sum())
    print(f'After local window: labeled {n_labeled2}/{N} ({100*n_labeled2/N:.3f}%)')

    # KDTree propagation for remaining
    remaining_idx = np.where(labels == 0)[0]
    if remaining_idx.size > 0:
        labeled_idx = np.where(labels > 0)[0]
        if labeled_idx.size > 0:
            tree = cKDTree(pts[labeled_idx])
            dists, idxs = tree.query(pts[remaining_idx], k=args.max_knn)
            if args.max_knn == 1:
                dists = np.asarray(dists)
                idxs = np.asarray(idxs)
            # assign if distance less than threshold
            assign_mask = dists <= args.knn_radius
            assigned = 0
            for i, ok in enumerate(assign_mask):
                if ok:
                    src = labeled_idx[int(idxs[i])]
                    labels[remaining_idx[i]] = labels[src]
                    assigned += 1
            print(f'KDTree assigned {assigned}/{remaining_idx.size} (radius {args.knn_radius})')
        else:
            print('No labeled points to propagate from via KDTree')

    total_labeled = int((labels > 0).sum())
    print(f'Final labeled {total_labeled}/{N} ({100*total_labeled/N:.3f}%)')

    # write outputs
    if args.out_np:
        np.save(args.out_np, labels)
        print('Saved labels numpy to', args.out_np)
    if args.out_ply:
        write_ply_with_labels(args.ply, args.out_ply, labels)
        print('Saved propagated PLY to', args.out_ply)


if __name__ == '__main__':
    main()
