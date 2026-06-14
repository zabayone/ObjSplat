#!/usr/bin/env python3
"""DEVA-based video instance segmentation for LayerPano3D.

Uses DEVA (Tracking Anything with DEVA) with automatic SAM prompting to
perform video instance segmentation on perspective frame sequences, tracking
objects across frames and propagating instance IDs to 3D coordinates.

Setup matches the original LabelGS automatic pipeline:
  - DEVA checkpoint : checkpoints/DEVA-propagation.pth
  - SAM  checkpoint : checkpoints/sam_vit_h_4b8939.pth   (vit_h)
  Both checkpoints are looked up via get_sam_model() from deva.ext.automatic_sam,
  which expects cfg["sam_checkpoint"] and cfg["sam_model_type"].
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Locate DEVA submodule.
DEVA_DIR = Path(__file__).parent.parent / "submodules" / "deva"
DEVA_AVAILABLE = False

if DEVA_DIR.exists():
    sys.path.insert(0, str(DEVA_DIR))
    try:
        from deva.inference.inference_core import DEVAInferenceCore
        from deva.inference.result_utils import ResultSaver
        from deva.model.network import DEVA as DEVAModel
        from deva.ext.automatic_sam import get_sam_model
        import deva.ext.automatic_processor as deva_automatic_processor
        from deva.ext.automatic_processor import process_frame_automatic
        DEVA_AVAILABLE = True
    except ImportError as e:
        print(f"[WARN] DEVA import failed: {e}")


# Default DEVA config. Runtime arguments can override these values.
_DEFAULT_CFG = {
    # Core model parameters (needed by DEVA model constructor)
    "key_dim": 64,
    "value_dim": 512,
    "pix_feat_dim": 512,
    "top_k": 30,
    "mem_every": 5,
    "chunk_size": -1,
    "size": 480,
    "disable_long_term": False,
    "temporal_setting": "online",
    "num_voting_frames": 3,
    "amp": False,
    "enable_long_term": True,
    "max_mid_term_frames": 10,
    "min_mid_term_frames": 5,
    "num_prototypes": 128,
    "max_long_term_elements": 10000,
    "enable_long_term_count_usage": True,
    # SAM settings - keys expected by DEVA ext/automatic_sam.py
    "SAM2_MODEL_CFG": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "SAM2_CHECKPOINT_PATH": "checkpoints/SAM 2.1 Hiera Large.pt",
    "SAM_ENCODER_VERSION": "vit_h",
    "SAM_CHECKPOINT_PATH": "checkpoints/sam_vit_h_4b8939.pth",
    "MOBILE_SAM_CHECKPOINT_PATH": "checkpoints/mobile_sam.pt",
    "SAM_NUM_POINTS_PER_SIDE": 32,
    "SAM_NUM_POINTS_PER_BATCH": 64,
    "SAM_PRED_IOU_THRESHOLD": 0.92,
    "SAM_OVERLAP_THRESHOLD": 0.8,
    # Optional GroundingDINO integration
    "use_grounding_dino": False,
    "GROUNDING_DINO_CHECKPOINT": "checkpoints/groundingdino_swinb_cogvlm.pth",
    # SAM variant: 'original' | 'mobile' | 'sam2'
    "sam_variant": "original",
    # Legacy aliases kept for compatibility with previous wrapper versions.
    "sam_model_type": "vit_h",
    "sam_checkpoint": "checkpoints/sam_vit_h_4b8939.pth",
    # Automatic processor knobs
    "detection_every": 5,          # re-detect every N frames
    "max_missed_detection_count": 10,
    "max_num_objects": 200,
    "suppress_small_objects": True,
    "sam_pred_iou_threshold": 0.92,
    "sam_stability_score_threshold": 0.95,
    "sam_stability_score_offset": 1.0,
    "mask_min_area": 1000,
    "mask_max_area": 0.8,
}


def _numeric_suffix_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return (int(digits) if digits else -1, stem)


class DEVAInstanceSegmenter:
    """Video instance segmentation using DEVA + automatic SAM prompting."""

    def __init__(
        self,
        deva_checkpoint: str = "checkpoints/DEVA-propagation.pth",
        sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth",
        device: str = "mps",
        temporal_setting: str = "online",
        cfg_overrides: Optional[dict] = None,
    ) -> None:
        self.deva_checkpoint = deva_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.device = torch.device(device)
        self.temporal_setting = temporal_setting
        self.deva: Optional[DEVAInferenceCore] = None
        self.sam_model = None
        self.cfg: dict = {}
        self.device_str = str(self.device)
        self.cfg_overrides = cfg_overrides or {}

        if not DEVA_AVAILABLE:
            print("[WARN] DEVA not available - instance segmentation will be skipped")
            return
        if not Path(deva_checkpoint).exists():
            print(f"[WARN] DEVA checkpoint not found: {deva_checkpoint}")
            print("  Download: https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases")
            return
        # Only require the legacy SAM checkpoint file when using the original/mobile SAM variant.
        sam_variant_override = self.cfg_overrides.get('sam_variant') if self.cfg_overrides else None
        if sam_variant_override in (None, 'original', 'mobile'):
            if not Path(sam_checkpoint).exists():
                print(f"[WARN] SAM checkpoint not found: {sam_checkpoint}")
                return
        self._load_models()

    # Model loading.
    def _load_models(self) -> None:
        try:
            cfg = dict(_DEFAULT_CFG)
            cfg["temporal_setting"] = self.temporal_setting
            cfg["sam_checkpoint"] = self.sam_checkpoint
            cfg["SAM_CHECKPOINT_PATH"] = self.sam_checkpoint
            # Honor explicit sam_variant if provided in overrides.
            if 'sam_variant' in self.cfg_overrides:
                cfg['sam_variant'] = self.cfg_overrides.get('sam_variant')
            if 'SAM2_CHECKPOINT_PATH' in self.cfg_overrides:
                cfg['SAM2_CHECKPOINT_PATH'] = self.cfg_overrides.get('SAM2_CHECKPOINT_PATH')
            if 'SAM2_MODEL_CFG' in self.cfg_overrides:
                cfg['SAM2_MODEL_CFG'] = self.cfg_overrides.get('SAM2_MODEL_CFG')
            if 'GROUNDING_DINO_CHECKPOINT' in self.cfg_overrides:
                cfg['GROUNDING_DINO_CHECKPOINT'] = self.cfg_overrides.get('GROUNDING_DINO_CHECKPOINT')
            if 'use_grounding_dino' in self.cfg_overrides:
                cfg['use_grounding_dino'] = bool(self.cfg_overrides.get('use_grounding_dino'))
            cfg["enable_long_term"] = not cfg.get("disable_long_term", False)
            for key, value in self.cfg_overrides.items():
                if value is not None and key in cfg:
                    cfg[key] = value

            # Load DEVA network weights using official API.
            model_weights = torch.load(self.deva_checkpoint, map_location=self.device)
            net = DEVAModel(cfg)
            net.load_weights(model_weights)
            net = net.to(self.device).eval()

            # Build inference core
            deva_core = DEVAInferenceCore(net, config=cfg)
            deva_core.next_voting_frame = cfg["num_voting_frames"] - 1
            deva_core.enabled_long_id()

            # MPS/CPU compatibility: DEVA demo helper hardcodes .cuda().
            # Monkey-patch automatic_processor.get_input_frame_for_deva to use our device.
            from deva.dataset.utils import im_normalization
            import torch.nn.functional as F

            device_str = self.device_str

            def _get_input_frame_for_deva_portable(image_np: np.ndarray, min_side: int) -> torch.Tensor:
                image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255
                image = im_normalization(image)
                if min_side > 0:
                    h, w = image_np.shape[:2]
                    scale = min_side / min(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    image = image.unsqueeze(0)
                    image = F.interpolate(image, (new_h, new_w), mode="bilinear", align_corners=False)[0]
                return image.to(device_str)

            deva_automatic_processor.get_input_frame_for_deva = _get_input_frame_for_deva_portable

            # Load SAM via DEVA helper (handles vit_h / vit_l / vit_b)
            sam = get_sam_model(cfg, str(self.device))

            if cfg.get('sam_variant') == 'sam2':
                sam2_checkpoint = cfg.get('SAM2_CHECKPOINT_PATH', self.sam_checkpoint)
                sam2_cfg = cfg.get('SAM2_MODEL_CFG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
                print(
                    "[OK] SAM configured as sam2 / "
                    f"{Path(sam2_checkpoint).name} "
                    f"(cfg: {sam2_cfg})"
                )
            else:
                print(
                    "[OK] SAM configured as "
                    f"{cfg['sam_variant']} / {cfg['SAM_ENCODER_VERSION']} "
                    f"from {cfg['SAM_CHECKPOINT_PATH']}"
                )

            self.deva = deva_core
            self.sam_model = sam
            self.cfg = cfg
            print("[OK] DEVA + SAM loaded successfully")
        except Exception as exc:
            print(f"[WARN] Failed to load DEVA/SAM models: {exc}")
            self.deva = None
            self.sam_model = None

    # Per-sequence inference.
    def segment_frame_sequence(
        self,
        frame_paths: List[Path],
        min_area: int = 2000,
    ) -> Dict[int, np.ndarray]:
        """Run DEVA+SAM on an ordered list of frame images.

        Returns
        -------
        Dict[frame_idx -> instance_map (H, W, int32)]
        """
        if self.deva is None or self.sam_model is None:
            print("[WARN] DEVA/SAM not loaded - cannot segment frames")
            return {}

        # Reset temporary frame buffer between sequences.
        if hasattr(self.deva, "clear_buffer"):
            self.deva.clear_buffer()

        frame_instances: Dict[int, np.ndarray] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="deva_result_") as temp_out:
                result_saver = ResultSaver(
                    temp_out,
                    None,
                    dataset="demo",
                    object_manager=self.deva.object_manager,
                )

                with torch.inference_mode():
                    for frame_idx, frame_path in enumerate(
                        tqdm(frame_paths, desc="DEVA segmentation")
                    ):
                        frame_np = np.array(
                            Image.open(frame_path).convert("RGB"), dtype=np.uint8
                        )
                        process_frame_automatic(
                            self.deva,
                            self.sam_model,
                            str(frame_path),
                            result_saver,
                            frame_idx,
                            image_np=frame_np,
                        )

                # Flush queued writes.
                result_saver.end()

                ann_dir = Path(temp_out) / "Annotations"
                for frame_idx, frame_path in enumerate(frame_paths):
                    mask_path = ann_dir / f"{frame_path.stem}.png"
                    if mask_path.exists():
                        raw = np.array(Image.open(mask_path))
                        if raw.ndim == 3 and raw.shape[2] == 3:
                            # DEVA demo saver writes RGB-encoded long IDs.
                            imap = (
                                np.round(raw[:, :, 0]).astype(np.int32)
                                + 256 * np.round(raw[:, :, 1]).astype(np.int32)
                                + 65536 * np.round(raw[:, :, 2]).astype(np.int32)
                            )
                        else:
                            imap = np.round(raw).astype(np.int32)
                    else:
                        frame_np = np.array(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
                        imap = np.zeros(frame_np.shape[:2], dtype=np.int32)
                    if min_area > 1:
                        # Remove tiny islands to reduce flicker and noise.
                        unique_ids = np.unique(imap)
                        for uid in unique_ids:
                            if uid == 0:
                                continue
                            area = int((imap == uid).sum())
                            if area < min_area:
                                imap[imap == uid] = 0
                    frame_instances[frame_idx] = imap
        except Exception as exc:
            import traceback
            print(f"[WARN] Error during DEVA segmentation: {exc}")
            traceback.print_exc()

        if hasattr(self.deva, "clear_buffer"):
            self.deva.clear_buffer()
        return frame_instances


# Module-level helpers.

def segment_perspective_frames(
    frames_dir: Path,
    deva_checkpoint: str = "checkpoints/DEVA-propagation.pth",
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth",
    device: str = "mps",
    temporal_setting: str = "online",
    min_area: int = 2000,
    sam_pred_iou_threshold: Optional[float] = None,
    sam_stability_score_threshold: Optional[float] = None,
    mask_min_area: Optional[int] = None,
    detection_every: Optional[int] = None,
    max_num_objects: Optional[int] = None,
    use_grounding_dino: bool = False,
    grounding_dino_checkpoint: Optional[str] = None,
    sam_variant: str = "original",
    sam2_checkpoint: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Segment all perspective frames in *frames_dir* using DEVA+SAM.

    Parameters
    ----------
    frames_dir      : directory containing rgb_*.png frames (from gen_traindata)
    deva_checkpoint : path to DEVA-propagation.pth
    sam_checkpoint  : path to sam_vit_h_4b8939.pth
    device          : 'mps' | 'cuda' | 'cpu'
    temporal_setting: 'online' or 'semionline'
    min_area        : minimum object area in pixels to keep

    Returns
    -------
    Dict[frame_idx -> instance_map (H, W, int32)]
    """
    if not frames_dir.exists():
        print(f"[WARN] Frames directory not found: {frames_dir}")
        return {}

    frame_paths = sorted(frames_dir.glob("rgb_*.png"), key=_numeric_suffix_key)
    if not frame_paths:
        # Fallback: any PNG
        frame_paths = sorted(frames_dir.glob("*.png"), key=_numeric_suffix_key) + sorted(
            frames_dir.glob("*.jpg"), key=_numeric_suffix_key
        )
    if not frame_paths:
        print(f"[WARN] No frames found in {frames_dir}")
        return {}

    print(f"  Found {len(frame_paths)} frames in {frames_dir.name}")

    cfg_overrides = {
        "sam_pred_iou_threshold": sam_pred_iou_threshold,
        "sam_stability_score_threshold": sam_stability_score_threshold,
        "mask_min_area": mask_min_area,
        "detection_every": detection_every,
        "max_num_objects": max_num_objects,
    }
    # Pass through new options
    cfg_overrides['use_grounding_dino'] = use_grounding_dino
    if grounding_dino_checkpoint is not None:
        cfg_overrides['GROUNDING_DINO_CHECKPOINT'] = grounding_dino_checkpoint
    if sam_variant is not None:
        cfg_overrides['sam_variant'] = sam_variant
    if sam2_checkpoint is not None:
        cfg_overrides['SAM2_CHECKPOINT_PATH'] = sam2_checkpoint
    segmenter = DEVAInstanceSegmenter(
        deva_checkpoint=deva_checkpoint,
        sam_checkpoint=sam_checkpoint,
        device=device,
        temporal_setting=temporal_setting,
        cfg_overrides=cfg_overrides,
    )
    return segmenter.segment_frame_sequence(frame_paths, min_area=min_area)


def propagate_frame_instances_to_3d(
    frames_dir: Path,
    instance_maps: Dict[int, np.ndarray],
    xyz: np.ndarray,
) -> np.ndarray:
    """Project 3D points into each perspective frame and vote on instance ID.

    Uses the transform_matrix_{i}.npy files saved by gen_traindata (4x4
    camera-to-world matrices, FOV=90°) and pinhole projection.

    Parameters
    ----------
    frames_dir      : directory containing transform_matrix_*.npy + rgb_*.png
    instance_maps   : Dict[frame_idx -> (H, W) int32 instance map]
    xyz             : (N, 3) world-space point cloud

    Returns
    -------
    (N,) int32 array of per-point instance IDs (0 = unlabelled)
    """
    import numpy.linalg as LA

    num_points = xyz.shape[0]
    votes: list[dict] = [{} for _ in range(num_points)]  # pt_idx -> {id: count}

    for frame_idx, fmap in instance_maps.items():
        transform_path = frames_dir / f"transform_matrix_{frame_idx}.npy"
        rgb_path = frames_dir / f"rgb_{frame_idx}.png"
        if not transform_path.exists() or not rgb_path.exists():
            continue

        try:
            pose_c2w = np.load(transform_path).astype(np.float64)   # 4x4 c2w
            img = Image.open(rgb_path)
            W, H = img.size
        except Exception:
            continue

        # World -> GS camera (invert c2w); convert to the perspective-frame convention
        # used by utils.pano_utils.Equirec2Perspec.GetPerspective:
        #   x = forward, y = right, z = up.
        w2c = LA.inv(pose_c2w)
        xyz_h = np.hstack([xyz, np.ones((num_points, 1), dtype=np.float64)])
        pts_gs = (w2c @ xyz_h.T).T[:, :3]   # (N, 3)
        pts_cam = np.stack(
            [pts_gs[:, 2], pts_gs[:, 0], -pts_gs[:, 1]],
            axis=1,
        )

        forward = pts_cam[:, 0]
        valid = forward > 1e-4  # in front of camera

        # Pinhole: FOV=90° -> focal = W/2
        focal = float(W) / 2.0
        finite = np.isfinite(pts_cam).all(axis=1)
        valid = valid & finite
        denom = np.where(valid, forward, 1.0)
        x_ndc = np.divide(pts_cam[:, 1], denom, out=np.zeros_like(forward), where=valid)
        y_ndc = np.divide(pts_cam[:, 2], denom, out=np.zeros_like(forward), where=valid)

        u_float = x_ndc * focal + float(W) / 2.0
        v_float = -y_ndc * focal + float(H) / 2.0
        finite_uv = np.isfinite(u_float) & np.isfinite(v_float)
        u = np.zeros(num_points, dtype=np.int32)
        v = np.zeros(num_points, dtype=np.int32)
        u[finite_uv] = u_float[finite_uv].astype(np.int32)
        v[finite_uv] = v_float[finite_uv].astype(np.int32)

        inside = valid & finite_uv & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        inds = np.nonzero(inside)[0]
        if inds.size == 0:
            continue

        labels_sampled = fmap[v[inds], u[inds]]
        for pt_idx, lab in zip(inds, labels_sampled):
            if lab == 0:
                continue
            d = votes[pt_idx]
            d[int(lab)] = d.get(int(lab), 0) + 1

    output = np.zeros(num_points, dtype=np.int32)
    for i, v in enumerate(votes):
        if v:
            output[i] = max(v.items(), key=lambda x: x[1])[0]
    return output
