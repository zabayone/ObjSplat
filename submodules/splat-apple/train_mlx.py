import datetime
import math
import os
import random
import argparse
from glob import glob

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import MLX GS modules
from mlx_gs.core.gaussians import Gaussians, init_gaussians_from_pcd
from mlx_gs.io.colmap import load_colmap_dataset, pose_to_w2c
from mlx_gs.training.trainer import Camera, train_step


def _logit(p):
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _build_optimizers():
    return {
        "means": optim.Adam(learning_rate=0.00016),
        "scales": optim.Adam(learning_rate=0.005),
        "quaternions": optim.Adam(learning_rate=0.001),
        "opacities": optim.Adam(learning_rate=0.05),
        "sh_coeffs": optim.Adam(learning_rate=0.0025),
    }


def _estimate_log_scales(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.full((points.shape[0], 3), -5.0, dtype=np.float32)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        nn = np.maximum(distances[:, 1] * 0.5, 1e-3)
        log_scale = np.log(nn).astype(np.float32)
        return np.repeat(log_scale[:, None], 3, axis=1)
    except Exception:
        return np.full((points.shape[0], 3), -5.0, dtype=np.float32)


def _resolve_layerpano_data(data_dir: str):
    frames_dir = os.path.join(data_dir, "frames")
    if not os.path.isdir(frames_dir):
        return None

    pcd_candidates = sorted(glob(os.path.join(data_dir, "pcd_rgb_layer*.ply")))
    if not pcd_candidates:
        pcd_candidates = sorted(glob(os.path.join(data_dir, "pcd_rgb.ply")))
    if not pcd_candidates:
        return None

    return {"frames_dir": frames_dir, "pcd_path": pcd_candidates[0]}


def _load_layerpano_dataset(data_dir: str):
    from plyfile import PlyData

    layer_info = _resolve_layerpano_data(data_dir)
    if layer_info is None:
        raise RuntimeError(f"{data_dir} does not look like a LayerPano layer folder")

    ply = PlyData.read(layer_info["pcd_path"])
    vertices = ply["vertex"].data
    xyz = np.stack([np.asarray(vertices[name], dtype=np.float32) for name in ("x", "y", "z")], axis=-1)

    if all(name in vertices.dtype.names for name in ("red", "green", "blue")):
        rgb = np.stack([np.asarray(vertices[name], dtype=np.float32) for name in ("red", "green", "blue")], axis=-1)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = np.full_like(xyz, 0.5, dtype=np.float32)

    frame_paths = sorted(
        glob(os.path.join(layer_info["frames_dir"], "rgb_*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
    )
    if not frame_paths:
        raise RuntimeError(f"No rgb_*.png frames found in {layer_info['frames_dir']}")

    frames = []
    for frame_path in frame_paths:
        frame_idx = int(os.path.splitext(os.path.basename(frame_path))[0].split("_")[-1])
        pose_path = os.path.join(layer_info["frames_dir"], f"transform_matrix_{frame_idx}.npy")
        if not os.path.exists(pose_path):
            continue
        frames.append({"image": Image.open(frame_path).convert("RGB"), "transform_matrix": np.load(pose_path)})

    if not frames:
        raise RuntimeError(f"No paired frame/pose files found in {layer_info['frames_dir']}")

    width, height = frames[0]["image"].size
    fov_deg = 90.0
    fovx = math.radians(fov_deg)
    fovy = height * fovx / width
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    cx = width / 2.0
    cy = height / 2.0

    cameras = []
    targets = []
    for frame in frames:
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        w2c = np.linalg.inv(pose).astype(np.float32)
        cameras.append(
            Camera(
                W=width,
                H=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                W2C=mx.array(w2c),
            )
        )

    return xyz, rgb, cameras, targets


def _adaptive_topology_update(params, prune_threshold=0.02, clone_fraction=0.08, min_points=128, scale_log_max: float = -2.0):
    means = np.array(params["means"])
    scales = np.array(params["scales"])
    quats = np.array(params["quaternions"])
    opacities = np.array(params["opacities"])
    sh_coeffs = np.array(params["sh_coeffs"])

    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities[:, 0]))
    score = opacity_sigmoid * np.exp(scales).mean(axis=1)

    keep_mask = opacity_sigmoid >= prune_threshold
    if keep_mask.sum() < min_points and len(score) > 0:
        keep_mask = np.zeros(len(score), dtype=bool)
        keep_mask[np.argsort(score)[-min(min_points, len(score)) :]] = True

    means = means[keep_mask]
    scales = scales[keep_mask]
    quats = quats[keep_mask]
    opacities = opacities[keep_mask]
    sh_coeffs = sh_coeffs[keep_mask]
    score = score[keep_mask]

    opacity_sigmoid_local = 1.0 / (1.0 + np.exp(-opacities[:, 0]))
    max_scale = np.exp(scales).max(axis=1)
    split_threshold = float(np.exp(float(scale_log_max) * 0.7))
    split_mask = (max_scale > split_threshold) & (opacity_sigmoid_local > prune_threshold)
    clone_mask = (max_scale <= split_threshold) & (opacity_sigmoid_local > prune_threshold)

    donor_means = means
    donor_scales = scales
    donor_quats = quats
    donor_opacities = opacities
    donor_sh = sh_coeffs
    donor_score = score

    split_idx = np.where(split_mask)[0]
    clone_candidates = np.where(clone_mask)[0]

    if split_idx.size > 0:
        split_means = means[split_idx]
        split_scales = np.clip(scales[split_idx] - np.float32(np.log(1.6)), -7.0, 0.1)
        split_quats = quats[split_idx]
        split_opac = opacities[split_idx]
        split_sh = sh_coeffs[split_idx]

        local_extent = np.exp(scales[split_idx]).max(axis=1, keepdims=True)
        direction = np.random.randn(*split_means.shape).astype(np.float32)
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.maximum(direction_norm, 1e-6)
        split_means2 = split_means + direction * (local_extent * 0.75)

        keep_after_split = ~split_mask
        means = np.concatenate([means[keep_after_split], split_means, split_means2], axis=0)
        scales = np.concatenate([scales[keep_after_split], split_scales, split_scales], axis=0)
        quats = np.concatenate([quats[keep_after_split], split_quats, split_quats], axis=0)
        opacities = np.concatenate([opacities[keep_after_split], split_opac, split_opac], axis=0)
        sh_coeffs = np.concatenate([sh_coeffs[keep_after_split], split_sh, split_sh], axis=0)
        score = np.concatenate([score[keep_after_split], score[split_idx], score[split_idx]], axis=0)

    clone_count = 0
    if clone_candidates.size > 0 and clone_fraction > 0:
        clone_count = int(max(1, round(clone_candidates.size * clone_fraction)))
    if clone_count > 0:
        donor_scores = donor_score[clone_candidates]
        clone_indices = clone_candidates[np.argsort(donor_scores)[-clone_count:]]
        noise_scale = np.maximum(np.exp(donor_scales[clone_indices]) * 0.18, 1e-3)
        clone_means = donor_means[clone_indices] + np.random.normal(0.0, noise_scale).astype(np.float32)
        clone_scales = np.clip(donor_scales[clone_indices] - np.float32(np.log(2.0)), -7.0, 0.1)
        clone_quats = donor_quats[clone_indices]
        clone_opacities = np.full_like(donor_opacities[clone_indices], _logit(0.02), dtype=np.float32)
        clone_sh = donor_sh[clone_indices]

        means = np.concatenate([means, clone_means], axis=0)
        scales = np.concatenate([scales, clone_scales], axis=0)
        quats = np.concatenate([quats, clone_quats], axis=0)
        opacities = np.concatenate([opacities, clone_opacities], axis=0)
        sh_coeffs = np.concatenate([sh_coeffs, clone_sh], axis=0)

    return {
        "means": mx.array(means, dtype=mx.float32),
        "scales": mx.array(scales, dtype=mx.float32),
        "quaternions": mx.array(quats, dtype=mx.float32),
        "opacities": mx.array(opacities, dtype=mx.float32),
        "sh_coeffs": mx.array(sh_coeffs, dtype=mx.float32),
    }


def save_ply(path, gaussians):
    """Export MLX Gaussians to a LayerPano-compatible PLY."""
    from plyfile import PlyData, PlyElement

    xyz = np.array(gaussians.means)
    normals = np.zeros_like(xyz)
    f_dc = np.array(gaussians.sh_coeffs[:, 0, :]).reshape(-1, 3)
    f_rest = np.zeros((xyz.shape[0], 45), dtype=np.float32)
    opacities = np.array(gaussians.opacities)
    scales = np.array(gaussians.scales)
    rot = np.array(gaussians.quaternions)

    finite_mask = np.isfinite(xyz).all(axis=1)
    finite_mask &= np.isfinite(f_dc).all(axis=1)
    finite_mask &= np.isfinite(opacities).all(axis=1)
    finite_mask &= np.isfinite(scales).all(axis=1)
    finite_mask &= np.isfinite(rot).all(axis=1)

    if not np.all(finite_mask):
        print(f"Filtering out {int((~finite_mask).sum())} non-finite Gaussians before export.")
        xyz = xyz[finite_mask]
        normals = normals[finite_mask]
        f_dc = f_dc[finite_mask]
        f_rest = f_rest[finite_mask]
        opacities = opacities[finite_mask]
        scales = scales[finite_mask]
        rot = rot[finite_mask]

    dtype_full = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ] + [(f"f_rest_{i}", "f4") for i in range(45)] + [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements["x"], elements["y"], elements["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements["nx"], elements["ny"], elements["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements["f_dc_0"], elements["f_dc_1"], elements["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(45):
        elements[f"f_rest_{i}"] = f_rest[:, i]
    elements["opacity"] = opacities[:, 0]
    elements["scale_0"], elements["scale_1"], elements["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    elements["rot_0"], elements["rot_1"], elements["rot_2"], elements["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def run_training(
    data_dir: str,
    img_folder: str = "images_8",
    num_iterations: int = 2000,
    rasterizer_type: str = "python",
    normalize_scene: bool = False,
    adaptive: bool = True,
    densify_interval: int = 120,
    prune_threshold: float = 0.02,
    clone_fraction: float = 0.08,
    scale_log_max: float = -2.0,
):
    """Generic MLX training loop with optional LayerPano layer-folder support."""
    print(f"Loading data from {data_dir}...")

    layerpano_data = None
    if _resolve_layerpano_data(data_dir) is not None:
        layerpano_data = _load_layerpano_dataset(data_dir)

    if layerpano_data is None:
        xyz, rgb, cameras, targets = load_colmap_dataset(data_dir, img_folder)
    else:
        xyz, rgb, cameras, targets = layerpano_data

    if normalize_scene:
        cam_centers = []
        for cam in cameras:
            w2c = np.array(cam.W2C)
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            cam_centers.append(-R.T @ T)
        cam_centers = np.array(cam_centers)
        centroid = np.mean(cam_centers, axis=0)
        avg_dist = np.mean(np.linalg.norm(cam_centers - centroid, axis=1))
        scale = 1.0 / (avg_dist + 1e-6)
        print(f"Normalizing scene: centroid={centroid}, scale={scale}")
        xyz = (xyz - centroid) * scale
        for cam in cameras:
            w2c = np.array(cam.W2C)
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            new_T = (R @ centroid + T) * scale
            new_w2c = np.eye(4)
            new_w2c[:3, :3] = R
            new_w2c[:3, 3] = new_T
            cam.W2C = mx.array(new_w2c, dtype=mx.float32)

    if np.all(rgb == 0):
        print("Detected zero colors in point cloud. Initializing with random colors.")
        rgb = np.random.uniform(0.4, 0.6, size=rgb.shape)

    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(cameras)} cameras")

    scale_init = _estimate_log_scales(np.asarray(xyz, dtype=np.float32))
    gaussians = init_gaussians_from_pcd(
        np.asarray(xyz, dtype=np.float32),
        np.asarray(rgb, dtype=np.float32),
        sh_degree=3,
        scale_init=scale_init,
        opacity_init=0.1,
    )

    params = {
        "means": gaussians.means,
        "scales": gaussians.scales,
        "quaternions": gaussians.quaternions,
        "opacities": gaussians.opacities,
        "sh_coeffs": gaussians.sh_coeffs,
    }
    optimizers = _build_optimizers()

    os.makedirs("results", exist_ok=True)
    dataset_name = os.path.basename(data_dir.rstrip("/"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "layerpano" if layerpano_data is not None else "colmap"
    output_dir = os.path.join("results", f"{dataset_name}_mlx_{mode_suffix}_{timestamp}")
    if rasterizer_type == "cpp":
        output_dir = os.path.join("results", f"{dataset_name}_mlx_{mode_suffix}_cpp_{timestamp}")

    progress_dir = os.path.join(output_dir, "progress")
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    pbar = tqdm(range(num_iterations))
    sh_warmup_iters = max(1, num_iterations // 4)
    for i in pbar:
        idx = random.randint(0, len(cameras) - 1)
        cam, target = cameras[idx], targets[idx]
        active_sh_degree = 0 if i < sh_warmup_iters else 1

        loss, rendered_image, psnr, _ = train_step(
            params,
            optimizers,
            target,
            cam,
            lambda_ssim=0.2,
            rasterizer_type=rasterizer_type,
            active_sh_degree=active_sh_degree,
        )

        mx.eval(params, loss, psnr)

        if adaptive and i > 0 and i % densify_interval == 0:
            params = _adaptive_topology_update(
                params,
                prune_threshold=prune_threshold,
                clone_fraction=clone_fraction,
                scale_log_max=scale_log_max,
            )
            optimizers = _build_optimizers()

        if i % 10 == 0:
            if mx.isnan(loss):
                print(f"\nIteration {i}: NaN detected in loss!")
                break

            pbar.set_description(f"Loss: {loss.item():.4f} PSNR: {psnr.item():.2f}")

            if i % 100 == 0:
                img_np = np.array(rendered_image)
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{i:04d}.png")
                )

    gaussians_final = Gaussians(**params)
    print("Training done. Saving final model...")
    save_ply(os.path.join(ply_dir, f"{dataset_name}_final.ply"), gaussians_final)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting on MLX")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--img_folder", type=str, default="images_8", help="Name of the image folder (e.g., images, images_4, images_8)")
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--rasterizer", type=str, default="python", choices=["python", "cpp"], help="Rasterizer version")
    parser.add_argument("--normalize", action="store_true", help="Apply scene normalization (recommended for 360 scenes)")
    parser.add_argument("--no_adaptive", action="store_true", help="Disable heuristic densification/pruning")
    parser.add_argument("--densify_interval", type=int, default=120, help="Iterations between adaptive topology updates")
    parser.add_argument("--prune_threshold", type=float, default=0.02, help="Opacity threshold for pruning")
    parser.add_argument("--clone_fraction", type=float, default=0.08, help="Fraction of surviving gaussians cloned at each update")
    parser.add_argument("--scale_log_max", type=float, default=-2.0, help="Upper bound for log-scales used by the split rule")

    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        img_folder=args.img_folder,
        num_iterations=args.num_iterations,
        rasterizer_type=args.rasterizer,
        normalize_scene=args.normalize,
        adaptive=not args.no_adaptive,
        densify_interval=args.densify_interval,
        prune_threshold=args.prune_threshold,
        clone_fraction=args.clone_fraction,
        scale_log_max=args.scale_log_max,
    )
