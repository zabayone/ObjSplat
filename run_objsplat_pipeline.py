#!/usr/bin/env python3
"""Run the object-aware layered 3DGS pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from LayerPano import LayerPano
from generate_layer_data import _generate_frames, generate_layer_data
from mps_splat_backend import (
    global_refine_after_merge,
    merge_ply_layers,
    mood_refine_after_merge,
)


def _load_metadata(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _existing_traindata_path(save_root: Path) -> Path:
    return save_root / "traindata"


def _has_existing_layers(save_root: Path) -> bool:
    traindata_dir = _existing_traindata_path(save_root)
    metadata_path = traindata_dir / "layer_instances.json"
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
    for name in ["layer_instances.json", "layer_mask_visualization.png"]:
        candidate = traindata_dir / name
        if candidate.exists():
            candidate.unlink()
    layer_instances_dir = traindata_dir / "layer_instances"
    if layer_instances_dir.exists():
        shutil.rmtree(layer_instances_dir)
    sky_dir = traindata_dir / "sky"
    if sky_dir.exists():
        shutil.rmtree(sky_dir)
    moods_dir = traindata_dir / "moods"
    if moods_dir.exists():
        shutil.rmtree(moods_dir)
    scene_dir = save_root / "scene"
    for name in [
        "gsplat_scene_night.ply",
        "gsplat_scene_active.ply",
        "moods.json",
    ]:
        candidate = scene_dir / name
        if candidate.is_symlink() or candidate.exists():
            candidate.unlink()


def _numeric_suffix_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return (int(digits) if digits else -1, stem)


def _load_full_scene_refine_frames(
    input_root: Path,
    max_image_size: int | None = None,
    frames_dir: Path | None = None,
) -> list[dict]:
    frames_dir = frames_dir or input_root / "traindata" / "perspective_frames" / "frames"
    frame_paths = sorted(frames_dir.glob("rgb_*.png"), key=_numeric_suffix_key)
    frames = []
    for rgb_path in frame_paths:
        idx, _stem = _numeric_suffix_key(rgb_path)
        pose_path = frames_dir / f"transform_matrix_{idx}.npy"
        if idx < 0 or not pose_path.exists():
            continue
        image = Image.open(rgb_path).convert("RGB")
        if max_image_size is not None and int(max_image_size) > 0:
            max_side = max(image.size)
            if max_side > int(max_image_size):
                scale = float(max_image_size) / float(max_side)
                image = image.resize(
                    (
                        max(1, int(round(image.size[0] * scale))),
                        max(1, int(round(image.size[1] * scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
        frames.append({
            "image": image,
            "transform_matrix": np.load(pose_path),
        })
    if not frames:
        raise RuntimeError(
            f"No full-scene refine frames found in {frames_dir}. "
            "Refine must use unmasked RGB views, not per-layer masked frames."
        )
    return frames


def _build_night_gaussian_mood(
    *,
    args,
    save_root: str,
    save_scene: str,
    metadata_path: Path,
    day_ply: Path,
    night_scene_erp: Path,
    night_mood_config,
) -> Path:
    from switch_mood import switch_mood
    from utils.mood_adaptation import (
        adapt_gaussian_ply_to_erp,
        write_mood_manifest,
    )

    metadata = _load_metadata(Path(metadata_path))
    sky_meta = metadata.get("sky") or {}
    sky_mask_path = Path(save_root) / str(
        sky_meta.get("mask_path", "traindata/sky/mask.png")
    )
    sky_radius = float(((sky_meta.get("geometry") or {}).get("radius") or 0.0))
    night_ply = Path(save_scene) / "gsplat_scene_night.ply"
    print("[pipeline] Fitting night ERP colors to the existing Gaussian topology")
    mood_fit_started = time.perf_counter()
    fit_report = adapt_gaussian_ply_to_erp(
        source_ply=day_ply,
        target_ply=night_ply,
        target_erp_path=night_scene_erp,
        sky_mask_path=sky_mask_path,
        config=night_mood_config,
        sky_radius=sky_radius,
    )
    print(
        f"[timing] night analytic fit elapsed="
        f"{time.perf_counter() - mood_fit_started:.1f}s"
    )

    if args.night_mood_refine_iters > 0:
        night_frames_dir = (
            Path(save_root) / "traindata" / "moods" / "night" / "frames"
        )
        if night_frames_dir.exists():
            shutil.rmtree(night_frames_dir)
        night_phi_bands = [
            float(value)
            for value in args.night_phi_bands.split(",")
            if value.strip()
        ]
        night_rgb = np.asarray(
            Image.open(night_scene_erp).convert("RGB"), dtype=np.uint8
        )
        _generate_frames(
            night_rgb,
            night_frames_dir,
            n=args.night_n_views,
            phi_bands=night_phi_bands,
            perspective_size=args.night_training_image_size,
        )
        mood_frames = _load_full_scene_refine_frames(
            Path(save_root),
            max_image_size=args.night_training_image_size,
            frames_dir=night_frames_dir,
        )
        first_w, first_h = mood_frames[0]["image"].size
        mood_traindata = {
            "fov": 90,
            "W": int(first_w),
            "H": int(first_h),
            "pcd_points": np.zeros((1, 3), dtype=np.float32),
            "pcd_colors": np.zeros((1, 3), dtype=np.float32),
            "pcd_masks": np.ones((1, 3), dtype=np.float32),
            "pcd_labels": np.zeros((1,), dtype=np.int32),
            "frames": mood_frames,
        }
        refined_night = night_ply.with_name(".gsplat_scene_night_refining.ply")
        mood_refine_after_merge(
            traindata=mood_traindata,
            initial_ply_path=str(night_ply),
            out_ply_path=str(refined_night),
            num_iterations=args.night_mood_refine_iters,
            rasterizer=args.mps_rasterizer,
            device=args.device,
        )
        os.replace(refined_night, night_ply)

    scene_root_path = Path(save_root).resolve()
    try:
        day_relative = str(day_ply.resolve().relative_to(scene_root_path))
        night_relative = str(night_ply.resolve().relative_to(scene_root_path))
    except ValueError as exc:
        raise ValueError("Mood PLY outputs must be inside the scene root") from exc
    write_mood_manifest(
        scene_root_path,
        {
            "active_mood": "day",
            "moods": {
                "day": {
                    "ply_path": day_relative,
                    "geometry_source": day_relative,
                },
                "night": {
                    "ply_path": night_relative,
                    "geometry_source": day_relative,
                    "scene_erp_path": str(
                        Path(night_scene_erp).resolve().relative_to(scene_root_path)
                    ),
                    "fit": fit_report,
                    "refine_iterations": int(args.night_mood_refine_iters),
                },
            },
        },
    )
    switch_mood(scene_root_path, "day")
    return night_ply


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
    parser.add_argument("--sam_checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--min_frame_area", type=int, default=2000)
    parser.add_argument("--min_frames", type=int, default=3)
    parser.add_argument("--min_total_pixels", type=int, default=10000)
    parser.add_argument("--min_points_3d", type=int, default=5000)
    parser.add_argument("--no_background", action="store_true")
    parser.add_argument("--force_resegment", action="store_true", help="Rerun layer generation even if traindata already exists")
    parser.add_argument("--segment_only", action="store_true", help="Stop after layer generation and overlay export")
    parser.add_argument(
        "--mood_only",
        action="store_true",
        help="Reuse an existing day PLY and only build the requested mood variant",
    )
    parser.add_argument("--refine_only", action="store_true", help="Skip segmentation/training/merge and refine an existing merged PLY")
    parser.add_argument("--skip_preflight", action="store_true",
                        help="Skip layer-data validation before diffusion/training")

    parser.add_argument("--outlier_thresh", type=int, default=3)
    parser.add_argument("--mps_rasterizer", type=str, default="cpp", choices=["python", "cpp"])
    parser.add_argument("--quality", type=str, default="standard", choices=["standard", "high", "ultra", "test"])
    parser.add_argument("--max_points", type=int, default=0,
                        help="Per-layer gaussian cap; <=0 disables the cap")
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument(
        "--training_image_size",
        type=int,
        default=512,
        help="Maximum side rasterized during 3DGS training; does not downsample points",
    )
    parser.add_argument("--layer_iterations", type=int, default=800)
    parser.add_argument("--background_iterations", type=int, default=1000)
    parser.add_argument("--sky_iterations", type=int, default=500)
    parser.add_argument("--repulsion_weight", type=float, default=1e-4)
    parser.add_argument("--mean_lr_scale", type=float, default=0.2)
    parser.add_argument(
        "--adaptive_topology",
        action="store_true",
        help="Enable prune/clone/split during layer training; disabled by default to preserve ERP coverage",
    )

    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--lr_plateau_patience", type=int, default=0)
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5)
    parser.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)

    parser.add_argument("--merge_voxel_size", type=float, default=0.0)
    parser.add_argument("--merge_min_opacity", type=float, default=-20.0,
                        help="Minimum stored opacity logit retained during merge")
    parser.add_argument("--merge_max_points", type=int, default=0,
                        help="Merged-scene gaussian cap; <=0 disables the cap")
    parser.add_argument("--merged_out", default=None)

    parser.add_argument("--global_refine_iters", type=int, default=0)
    parser.add_argument("--global_refine_layer", type=int, default=None)
    
    parser.add_argument("--frames_dir", default=None, help="Use an existing frames directory")
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--phi_bands", default="45,0,-45", help="Comma-separated phi bands in degrees")
    parser.add_argument("--perspective_size", type=int, default=1024,
                        help="Maximum square resolution of generated perspective training views")
    parser.add_argument("--use_grounding_dino", action="store_true", help="Enable GroundingDINO proposals + tagging")
    parser.add_argument("--grounding_dino_checkpoint", default="IDEA-Research/grounding-dino-base")
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
    parser.add_argument("--fill_unassigned_layers", action="store_true",
                        help="Legacy mode: force every ERP pixel into the nearest detected layer")
    parser.add_argument("--require_sky_layer", action="store_true",
                        help="Fail early unless a dedicated semantic sky layer is generated")
    parser.add_argument("--sky_segmentation_backend", default="grounding_sam",
                        choices=["grounding_sam", "hybrid", "segformer"])
    parser.add_argument("--sky_segformer_model",
                        default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--sky_segformer_max_side", type=int, default=2048)
    parser.add_argument("--sky_segformer_threshold", type=float, default=0.45)
    parser.add_argument("--sky_sphere_radius", type=float, default=0.0,
                        help="Explicit sky sphere radius; <=0 derives it from scene depth")
    parser.add_argument("--sky_radius_percentile", type=float, default=95.0)
    parser.add_argument("--sky_radius_scale", type=float, default=1.25)
    parser.add_argument("--retexture_night_sky", action="store_true",
                        help="Generate a night sky ERP with the local FLUX Fill checkpoint")
    parser.add_argument("--sky_model_path", default="checkpoints/FLUX.1-Fill-dev")
    parser.add_argument("--sky_prompt", default=None)
    parser.add_argument("--sky_seed", type=int, default=42)
    parser.add_argument("--sky_steps", type=int, default=50)
    parser.add_argument("--sky_guidance_scale", type=float, default=30.0)
    parser.add_argument("--sky_max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--sky_mask_dilate_px", type=int, default=5)
    parser.add_argument("--sky_mask_feather_px", type=int, default=9)
    parser.add_argument("--sky_circular_padding_ratio", type=float, default=0.0625)
    parser.add_argument("--sky_min_coverage", type=float, default=0.005)
    parser.add_argument("--sky_device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--sky_no_cpu_offload", action="store_true")
    parser.add_argument(
        "--build_night_mood",
        action="store_true",
        help="Relight non-sky regions and build a switchable night Gaussian PLY",
    )
    parser.add_argument("--night_exposure_ev", type=float, default=-2.65)
    parser.add_argument("--night_contrast", type=float, default=0.98)
    parser.add_argument("--night_saturation", type=float, default=0.32)
    parser.add_argument("--night_shadow_suppression", type=float, default=0.82)
    parser.add_argument("--night_shadow_blur_fraction", type=float, default=0.035)
    parser.add_argument(
        "--night_mood_refine_iters",
        type=int,
        default=0,
        help="Appearance-only night refinement iterations; 0 uses analytic SH fitting only",
    )
    parser.add_argument("--night_training_image_size", type=int, default=384)
    parser.add_argument("--night_n_views", type=int, default=8)
    parser.add_argument("--night_phi_bands", default="67.5,45,0,-45,-67.5")
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
    metadata_path = traindata_dir / "layer_instances.json"
    merged_out = args.merged_out
    if merged_out is None:
        merged_out = os.path.join(save_scene, "gsplat_scene_merged.ply")

    merge_max_points = args.merge_max_points if args.merge_max_points and args.merge_max_points > 0 else None

    if args.refine_only:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for refine_only: {metadata_path}")
        if not os.path.exists(merged_out):
            raise FileNotFoundError(f"Missing merged PLY for refine_only: {merged_out}")
        if not args.global_refine_iters or args.global_refine_iters <= 0:
            raise ValueError("--refine_only requires --global_refine_iters > 0")
        refine_frames = _load_full_scene_refine_frames(
            Path(save_root), max_image_size=args.training_image_size
        )
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
        print(f"[pipeline] Refine-only using {len(refine_frames)} full-scene frames.")
        refined_out = os.path.splitext(merged_out)[0] + "_refined.ply"
        global_refine_after_merge(
            traindata=traindata,
            merged_ply_path=merged_out,
            out_ply_path=refined_out,
            num_iterations=args.global_refine_iters,
            rasterizer=args.mps_rasterizer,
            device=args.device,
            adaptive=False,
            max_points=merge_max_points,
            downsample_ratio=args.downsample_ratio,
            repulsion_weight=args.repulsion_weight,
        )
        return

    has_existing_layers = _has_existing_layers(Path(save_root))
    if has_existing_layers and not args.force_resegment:
        print(f"[pipeline] Existing traindata found at {traindata_dir}; skipping layer generation.")
    else:
        if args.force_resegment and has_existing_layers:
            print(f"[pipeline] Clearing existing layer outputs at {traindata_dir} before resegmenting.")
            _clear_existing_layer_outputs(Path(save_root))
        metadata_path = Path(
            generate_layer_data(
                input_dir=args.input_dir,
                save_dir=save_root,
                depth_model=args.depth_model,
                depth_scale=args.depth_scale,
                auto_depth_scale=args.auto_depth_scale,
                target_scene_scale=args.target_scene_scale,
                device=args.device,
                sam_checkpoint=args.sam_checkpoint,
                min_frame_area=args.min_frame_area,
                min_frames=args.min_frames,
                min_total_pixels=args.min_total_pixels,
                min_points_3d=args.min_points_3d,
                add_background=not args.no_background,
                frames_dir=args.frames_dir,
                n_views=args.n_views,
                phi_bands=phi_bands,
                perspective_size=args.perspective_size,
                use_grounding_dino=args.use_grounding_dino,
                grounding_dino_checkpoint=args.grounding_dino_checkpoint,
                sam_variant=args.sam_variant,
                sam2_checkpoint=args.sam2_checkpoint,
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
                fill_unassigned_layers=args.fill_unassigned_layers,
                require_sky_layer=args.require_sky_layer,
                sky_segmentation_backend=args.sky_segmentation_backend,
                sky_segformer_model=args.sky_segformer_model,
                sky_segformer_max_side=args.sky_segformer_max_side,
                sky_segformer_threshold=args.sky_segformer_threshold,
                sky_sphere_radius=args.sky_sphere_radius,
                sky_radius_percentile=args.sky_radius_percentile,
                sky_radius_scale=args.sky_radius_scale,
                use_full_scene_background=args.use_full_scene_background,
                equirect_min_votes=args.equirect_min_votes,
                equirect_kernel_size=args.equirect_kernel_size,
            )
        )

    if not args.skip_preflight:
        from utils.pipeline_validation import assert_valid_layer_data

        assert_valid_layer_data(
            save_root,
            metadata_path=metadata_path,
            require_sky=bool(
                args.require_sky_layer
                or args.retexture_night_sky
                or args.build_night_mood
            ),
            min_sky_coverage=args.sky_min_coverage,
        )

    if args.retexture_night_sky:
        from utils.sky_retexture import DEFAULT_NIGHT_PROMPT, SkyRetextureConfig, retexture_sky

        print("[pipeline] Generating masked night-sky ERP")
        retexture_sky(
            scene_root=save_root,
            metadata_path=metadata_path,
            config=SkyRetextureConfig(
                model_path=args.sky_model_path,
                prompt=args.sky_prompt or DEFAULT_NIGHT_PROMPT,
                seed=args.sky_seed,
                num_inference_steps=args.sky_steps,
                guidance_scale=args.sky_guidance_scale,
                max_pixels=args.sky_max_pixels,
                mask_dilate_px=args.sky_mask_dilate_px,
                mask_feather_px=args.sky_mask_feather_px,
                circular_padding_ratio=args.sky_circular_padding_ratio,
                min_sky_coverage=args.sky_min_coverage,
                device=args.sky_device,
                cpu_offload=not args.sky_no_cpu_offload,
            ),
        )

    night_mood_config = None
    night_scene_erp = None
    if args.build_night_mood:
        from utils.mood_adaptation import NightMoodConfig, build_night_scene_erp

        night_mood_config = NightMoodConfig(
            exposure_ev=args.night_exposure_ev,
            contrast=args.night_contrast,
            saturation=args.night_saturation,
            shadow_suppression=args.night_shadow_suppression,
            shadow_blur_fraction=args.night_shadow_blur_fraction,
        )
        print("[pipeline] Relighting the non-sky scene for the night mood")
        night_scene_erp = build_night_scene_erp(
            scene_root=save_root,
            metadata_path=metadata_path,
            config=night_mood_config,
        )

    if args.mood_only:
        if not args.build_night_mood:
            raise ValueError("--mood_only requires --build_night_mood")
        refined_candidate = Path(os.path.splitext(merged_out)[0] + "_refined.ply")
        day_candidate = refined_candidate if refined_candidate.exists() else Path(merged_out)
        if not day_candidate.exists():
            raise FileNotFoundError(
                f"No existing day PLY for mood-only mode: {day_candidate}"
            )
        _build_night_gaussian_mood(
            args=args,
            save_root=save_root,
            save_scene=save_scene,
            metadata_path=Path(metadata_path),
            day_ply=day_candidate,
            night_scene_erp=Path(night_scene_erp),
            night_mood_config=night_mood_config,
        )
        return

    if args.segment_only:
        print("[pipeline] segment_only enabled; skipping layer training and merge.")
        return

    layerpano = LayerPano(
        save_dir=save_scene,
        backend="splat-apple",
        mps_rasterizer=args.mps_rasterizer,
        quality=args.quality,
        max_points=args.max_points,
        downsample_ratio=args.downsample_ratio,
        training_image_size=args.training_image_size,
        layer_iterations=args.layer_iterations,
        background_iterations=args.background_iterations,
        sky_iterations=args.sky_iterations,
        disable_transfer=True,
        no_adaptive=not args.adaptive_topology,
        repulsion_weight=args.repulsion_weight,
        mean_lr_scale=args.mean_lr_scale,
        early_stop_patience=(args.early_stop_patience or None),
        early_stop_min_delta=args.early_stop_min_delta,
        lr_plateau_patience=(args.lr_plateau_patience or None),
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_min_lr=args.lr_plateau_min_lr,
        mode="layer_instances",
    )

    ply_paths = layerpano.create_layer_instances(
        args.input_dir,
        outlier_thresh=args.outlier_thresh,
        metadata_path=str(metadata_path),
        background_last=True,
    )

    if not ply_paths:
        raise RuntimeError("No layer PLYs produced")

    merge_started = time.perf_counter()
    merge_ply_layers(
        ply_paths,
        merged_out,
        voxel_size=args.merge_voxel_size,
        min_opacity=args.merge_min_opacity,
        max_points=merge_max_points,
    )
    print(f"[timing] merge elapsed={time.perf_counter() - merge_started:.1f}s")

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

        refine_frames = _load_full_scene_refine_frames(
            Path(save_root), max_image_size=args.training_image_size
        )
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
            adaptive=False,
            max_points=merge_max_points,
            downsample_ratio=args.downsample_ratio,
            repulsion_weight=args.repulsion_weight,
        )
        day_ply = Path(refined_out)
    else:
        day_ply = Path(merged_out)

    if args.build_night_mood:
        if night_mood_config is None or night_scene_erp is None:
            raise RuntimeError("Night mood ERP was not initialized")
        _build_night_gaussian_mood(
            args=args,
            save_root=save_root,
            save_scene=save_scene,
            metadata_path=Path(metadata_path),
            day_ply=day_ply,
            night_scene_erp=Path(night_scene_erp),
            night_mood_config=night_mood_config,
        )


if __name__ == "__main__":
    main()
