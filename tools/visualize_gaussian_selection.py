#!/usr/bin/env python3
"""Visualizza/esporta gaussiane riattivate prima del training.

Esempio:
  python tools/visualize_gaussian_selection.py --frozen_ply prev_layer.ply --pcd ply_current_layer.ply --out selected

Produzione:
  selected_gaussians.ply  (colorate rosso=selezionate, grigio=non)
  selected_indices.txt    (elenco indici selezionati)
"""
import sys
from pathlib import Path
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

# Ensure repo root is importable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mps_splat_backend import extract_gaussian_params_from_ply, _gaussian_selector


def load_ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    v = ply['vertex']
    return np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=1).astype(np.float32)


def write_ply_xyz_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    assert xyz.shape[0] == rgb.shape[0]
    vertex = []
    for i in range(xyz.shape[0]):
        x, y, z = float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])
        r, g, b = int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2])
        vertex.append((x, y, z, r, g, b))
    vertex_np = np.array(vertex, dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    el = PlyElement.describe(vertex_np, 'vertex')
    PlyData([el]).write(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frozen_ply', required=True)
    p.add_argument('--pcd', required=True, help='Current layer point cloud (PLY) used as new_points')
    p.add_argument('--out', default='selected')
    p.add_argument('--beta3', type=float, default=10.0)
    args = p.parse_args()

    frozen_ply = Path(args.frozen_ply)
    pcd = Path(args.pcd)
    out_prefix = Path(args.out)

    params, labels = extract_gaussian_params_from_ply(str(frozen_ply))
    if params is None:
        print('Failed to extract gaussian params from', frozen_ply)
        return

    frozen_means = params['means']
    frozen_scales = params['scales']

    new_xyz = load_ply_xyz(pcd)

    mask = _gaussian_selector(frozen_means, frozen_scales, new_xyz, beta3=args.beta3)

    # Colors: selected -> red, others -> gray
    rgb = np.tile(np.array([180,180,180], dtype=np.uint8)[None,:], (frozen_means.shape[0],1))
    rgb[mask] = np.array([230,30,30], dtype=np.uint8)

    out_ply = out_prefix.with_name(out_prefix.name + '_gaussians_selected.ply')
    write_ply_xyz_rgb(out_ply, frozen_means, rgb)

    out_idx = out_prefix.with_name(out_prefix.name + '_selected_indices.txt')
    np.savetxt(str(out_idx), np.where(mask)[0].astype(np.int32), fmt='%d')

    print(f'Wrote {out_ply} and {out_idx} (selected {mask.sum()}/{len(mask)})')


if __name__ == '__main__':
    main()
