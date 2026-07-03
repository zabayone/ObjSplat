#!/usr/bin/env python3
"""Run the object-aware layered 3DGS pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from LayerPano import LayerPano
from gen_layerdata_from_deva import generate_layers_from_deva
from mps_splat_backend import merge_ply_layers, global_refine_after_merge


def _load_metadata(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _existing_traindata_path(save_root: Path) -> Path:
    return save_root / "traindata"


def _has_existing_layers(save_root: Path) -> bool:
    traindata_dir = _existing_traindata_path(save_root)
    metadata_path = traindata_dir / "deva_instances.json"
    if not metadata_path.exists():
        return False
    for layer_dir in traindata_dir.glob("layer*/"):
        if any(layer_dir.glob("pcd_rgb_layer*.ply")) and any((layer_dir / "frames").glob("rgb_*.png")):
            return True
    return False


def _clear_existing_layer_outputs(save_root: Path) -> None:
    traindata_dir = _existing_traindata_path(save_root)
    if not traindata_dir.exists():
        return
    for layer_dir in traindata_dir.glob("layer*"):
        if layer_dir.is_dir():
            shutil.rmtree(layer_dir)
    for name in ["deva_instances.json", "layer_mask_visualization.png"]:
        candidate = traindata_dir / name
        if candidate.exists():
            candidate.unlink()
    deva_instances_dir = traindata_dir / "deva_instances"
    if deva_instances_dir.exists():
        shutil.rmtree(deva_instances_dir)


def _numeric_suffix_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return (int(digits) if digits else -1, stem)


def _load_full_scene_refine_frames(input_root: Path) -> list[dict]:
    frames_dir = input_root / "traindata" / "deva_frames" / "frames"
    frame_paths = sorted(frames_dir.glob("rgb_*.png"), key=_numeric_suffix_key)
    frames = []
    for rgb_path in frame_paths:
        idx, _stem = _numeric_suffix_key(rgb_path)
        pose_path = frames_dir / f"transform_matrix_{idx}.npy"
        if idx < 0 or not pose_path.exists():
            continue
        frames.append({
            "image": Image.open(rgb_path).convert("RGB"),
            "transform_matrix": np.load(pose_path),
        })
    if not frames:
        raise RuntimeError(
            f"No full-scene refine frames found in {frames_dir}. "
            "Refine must use unmasked RGB views, not per-layer masked frames."
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Object-aware layered 3DGS training pipeline")
    parser.add_argument("--input_dir", required=True, help="Input directory (e.g. outputs_lgs)")
    parser.add_argument("--save_dir", default=None, help="Output root (defaults to input_dir)")
    parser.add_argument("--depth_model", default="DepthAnythingv2")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--auto_depth_scale", action="store_true",
                        help="Automatically scale depth to match target scene size")
    parser.add_argument("--target_scene_scale", type=float, default=0.5,
                        help="Target scene std (used when --auto_depth_scale enabled)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--deva_checkpoint", default="checkpoints/DEVA-propagation.pth")
    parser.add_argument("--sam_checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--min_frame_area", type=int, default=2000)
    parser.add_argument("--min_frames", type=int, default=3)
    parser.add_argument("--min_total_pixels", type=int, default=10000)
    parser.add_argument("--min_points_3d", type=int, default=5000)
    parser.add_argument("--no_background", action="store_true")
    parser.add_argument("--force_resegment", action="store_true", help="Rerun layer generation even if traindata already exists")
    parser.add_argument("--segment_only", action="store_true", help="Stop after layer generation and overlay export")

    parser.add_argument("--outlier_thresh", type=int, default=3)
    parser.add_argument("--mps_rasterizer", type=str, default="cpp", choices=["python", "cpp"])
    parser.add_argument("--mps_training_backend", type=str, default="mlx", choices=["auto", "torch", "mlx"])
    parser.add_argument("--quality", type=str, default="standard", choices=["standard", "high", "ultra", "test"])
    parser.add_argument("--max_points", type=int, default=0,
                        help="Per-layer gaussian cap; <=0 disables the explicit cap")
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--repulsion_weight", type=float, default=1e-4)
    parser.add_argument("--mean_lr_scale", type=float, default=0.2)

    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--lr_plateau_patience", type=int, default=0)
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5)
    parser.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)

    parser.add_argument("--merge_voxel_size", type=float, default=0.0)
    parser.add_argument("--merge_min_opacity", type=float, default=-20.0)
    parser.add_argument("--merge_max_points", type=int, default=0,
                        help="Merged-scene gaussian cap; <=0 disables the cap")
    parser.add_argument("--merged_out", default=None)

    parser.add_argument("--global_refine_iters", type=int, default=300)
    parser.add_argument("--global_refine_layer", type=int, default=None)
    
    parser.add_argument("--frames_dir", default=None, help="Use an existing frames directory")
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--phi_bands", default="45,0,-45", help="Comma-separated phi bands in degrees")
    parser.add_argument("--temporal_setting", default="online", choices=["online", "semionline"])
    parser.add_argument("--sam_pred_iou_threshold", type=float, default=0.88)
    parser.add_argument("--sam_stability_score_threshold", type=float, default=0.95)
    parser.add_argument("--mask_min_area", type=int, default=1500)
    parser.add_argument("--detection_every", type=int, default=None)
    parser.add_argument("--max_num_objects", type=int, default=None)
    parser.add_argument("--use_grounding_dino", action="store_true", help="Enable GroundingDINO proposals + tagging")
    parser.add_argument("--grounding_dino_checkpoint", default="checkpoints/groundingdino_swinb_cogvlm.pth")
    parser.add_argument("--grounding_first", action="store_true",
                        help="Run GroundingDINO on the panorama, then SAM per box")
    parser.add_argument("--grounding_prompts", default=None,
                        help="GroundingDINO prompt string, e.g. 'person . chair . table'")
    parser.add_argument("--grounding_box_threshold", type=float, default=0.25)
    parser.add_argument("--grounding_text_threshold", type=float, default=0.20)
    parser.add_argument("--grounding_max_detections", type=int, default=None)
    parser.add_argument("--grounding_mask_min_area", type=int, default=1500)
    parser.add_argument("--grounding_single_mask", action="store_true",
                        help="Use SAM's best single mask per GroundingDINO box")
    parser.add_argument("--grounding_box_padding", type=float, default=0.15,
                        help="Padding ratio used to clip SAM masks around each GroundingDINO box")
    parser.add_argument("--grounding_infer_max_side", type=int, default=1024,
                        help="Max panorama side used for GroundingDINO inference")
    parser.add_argument("--grounding_exclude_labels", default=None,
                        help="Optional comma-separated labels to exclude from object layers")
    parser.add_argument("--grounding_min_component_area_ratio", type=float, default=0.02,
                        help="Drop detached SAM mask components smaller than this fraction of total mask area")
    parser.add_argument("--grounding_morph_open_kernel", type=int, default=5,
                        help="Opening kernel used to remove thin detached mask artifacts; set 0 to disable")
    parser.add_argument("--aggregate_by_label", action="store_true",
                        help="Group same-label Grounding-SAM instances into shared training layers while preserving instance labels")
    parser.add_argument("--sam_variant", default="sam2", choices=["original", "mobile", "sam2"])
    parser.add_argument("--sam2_checkpoint", default="checkpoints/SAM 2.1 Hiera Large.pt")
    
    parser.add_argument("--use_full_scene_background", action="store_true")
    parser.add_argument("--equirect_min_votes", type=int, default=2)
    parser.add_argument("--equirect_kernel_size", type=int, default=7)

    args = parser.parse_args()

    save_root = args.save_dir if args.save_dir else args.input_dir
    save_scene = os.path.join(save_root, "scene")
    os.makedirs(save_scene, exist_ok=True)

    phi_bands = [float(x) for x in args.phi_bands.split(",") if x.strip()]

    traindata_dir = _existing_traindata_path(Path(save_root))
    metadata_path = traindata_dir / "deva_instances.json"
    has_existing_layers = _has_existing_layers(Path(save_root))
    if has_existing_layers and not args.force_resegment:
        print(f"[pipeline] Existing traindata found at {traindata_dir}; skipping layer generation.")
    else:
        if args.force_resegment and has_existing_layers:
            print(f"[pipeline] Clearing existing layer outputs at {traindata_dir} before resegmenting.")
            _clear_existing_layer_outputs(Path(save_root))
        metadata_path = Path(
            generate_layers_from_deva(
                input_dir=args.input_dir,
                save_dir=save_root,
                depth_model=args.depth_model,
                depth_scale=args.depth_scale,
                auto_depth_scale=args.auto_depth_scale,
                target_scene_scale=args.target_scene_scale,
                device=args.device,
                deva_checkpoint=args.deva_checkpoint,
                sam_checkpoint=args.sam_checkpoint,
                min_frame_area=args.min_frame_area,
                min_frames=args.min_frames,
                min_total_pixels=args.min_total_pixels,
                min_points_3d=args.min_points_3d,
                add_background=not args.no_background,
                frames_dir=args.frames_dir,
                n_views=args.n_views,
                phi_bands=phi_bands,
                temporal_setting=args.temporal_setting,
                sam_pred_iou_threshold=args.sam_pred_iou_threshold,
                sam_stability_score_threshold=args.sam_stability_score_threshold,
                mask_min_area=args.mask_min_area,
                detection_every=args.detection_every,
                max_num_objects=args.max_num_objects,
                use_grounding_dino=args.use_grounding_dino,
                grounding_dino_checkpoint=args.grounding_dino_checkpoint,
                sam_variant=args.sam_variant,
                sam2_checkpoint=args.sam2_checkpoint,
                grounding_first=args.grounding_first,
                grounding_prompts=args.grounding_prompts,
                grounding_box_threshold=args.grounding_box_threshold,
                grounding_text_threshold=args.grounding_text_threshold,
                grounding_max_detections=args.grounding_max_detections,
                grounding_mask_min_area=args.grounding_mask_min_area,
                grounding_sam_multimask=not args.grounding_single_mask,
                grounding_box_padding=args.grounding_box_padding,
                grounding_infer_max_side=args.grounding_infer_max_side,
                grounding_exclude_labels=args.grounding_exclude_labels,
                grounding_min_component_area_ratio=args.grounding_min_component_area_ratio,
                grounding_morph_open_kernel=args.grounding_morph_open_kernel,
                aggregate_by_label=args.aggregate_by_label,
                use_full_scene_background=args.use_full_scene_background,
                equirect_min_votes=args.equirect_min_votes,
                equirect_kernel_size=args.equirect_kernel_size,
            )
        )

    if args.segment_only:
        print("[pipeline] segment_only enabled; skipping layer training and merge.")
        return

    layerpano = LayerPano(
        save_dir=save_scene,
        backend="splat-apple",
        mps_rasterizer=args.mps_rasterizer,
        quality=args.quality,
        mps_training_backend=args.mps_training_backend,
        max_points=args.max_points,
        downsample_ratio=args.downsample_ratio,
        disable_transfer=True,
        no_adaptive=False,
        repulsion_weight=args.repulsion_weight,
        mean_lr_scale=args.mean_lr_scale,
        early_stop_patience=(args.early_stop_patience or None),
        early_stop_min_delta=args.early_stop_min_delta,
        lr_plateau_patience=(args.lr_plateau_patience or None),
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_min_lr=args.lr_plateau_min_lr,
        mode="deva_instances",
    )

    ply_paths = layerpano.create_deva_instances(
        args.input_dir,
        outlier_thresh=args.outlier_thresh,
        metadata_path=str(metadata_path),
        background_last=True,
    )

    if not ply_paths:
        raise RuntimeError("No layer PLYs produced")

    merged_out = args.merged_out
    if merged_out is None:
        merged_out = os.path.join(save_scene, "gsplat_scene_merged.ply")

    merge_max_points = args.merge_max_points if args.merge_max_points and args.merge_max_points > 0 else None

    merge_ply_layers(
        ply_paths,
        merged_out,
        voxel_size=args.merge_voxel_size,
        min_opacity=args.merge_min_opacity,
        max_points=merge_max_points,
    )

    if args.global_refine_iters and args.global_refine_iters > 0:
        meta = _load_metadata(Path(metadata_path))
        layer_idx = args.global_refine_layer
        if layer_idx is None:
            layer_idx = meta.get("final_refine_layer_idx")
        if layer_idx is None:
            layer_idx = meta.get("background_layer_idx")
        if layer_idx is None:
            instances = meta.get("instances", [])
            if instances:
                layer_idx = instances[0].get("layer_idx")

        if layer_idx is None:
            raise RuntimeError("Could not resolve a layer for global refinement")

        refine_frames = _load_full_scene_refine_frames(Path(args.input_dir))
        first_w, first_h = refine_frames[0]["image"].size
        traindata = {
            "fov": 90,
            "W": int(first_w),
            "H": int(first_h),
            "pcd_points": np.zeros((1, 3), dtype=np.float32),
            "pcd_colors": np.zeros((1, 3), dtype=np.float32),
            "pcd_masks": np.ones((1, 3), dtype=np.float32),
            "pcd_labels": np.zeros((1,), dtype=np.int32),
            "frames": refine_frames,
        }
        print(f"[pipeline] Global refine using {len(refine_frames)} full-scene frames.")
        refined_out = os.path.splitext(merged_out)[0] + "_refined.ply"
        global_refine_after_merge(
            traindata=traindata,
            merged_ply_path=merged_out,
            out_ply_path=refined_out,
            num_iterations=args.global_refine_iters,
            rasterizer=args.mps_rasterizer,
            device=args.device,
            training_backend=args.mps_training_backend,
            adaptive=False,
            max_points=merge_max_points,
            downsample_ratio=args.downsample_ratio,
            repulsion_weight=args.repulsion_weight,
        )


if __name__ == "__main__":
    main()
