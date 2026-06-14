"""SAM / SAM2 automatic segmentation helpers used by DEVA.

This module now supports:
- original SAM
- MobileSAM
- SAM2.1 Hiera variants via the `sam2` package
- optional GroundingDINO proposals used to seed SAM2 box prompts
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from deva.ext.SAM.automatic_mask_generator import SamAutomaticMaskGenerator
from deva.inference.object_info import ObjectInfo
from segment_anything import sam_model_registry

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except Exception:
    SAM2_AVAILABLE = False
    build_sam2 = None  # type: ignore[assignment]
    SAM2AutomaticMaskGenerator = None  # type: ignore[assignment]
    SAM2ImagePredictor = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    GROUNDING_DINO_AVAILABLE = True
except Exception:
    GROUNDING_DINO_AVAILABLE = False
    AutoModelForZeroShotObjectDetection = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]


def _resolve_sam2_config(checkpoint_path: str, config_path: Optional[str] = None) -> str:
    if config_path:
        return config_path

    lower_name = Path(checkpoint_path).name.lower()
    if "tiny" in lower_name:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    if "small" in lower_name:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    if "base" in lower_name or "b+" in lower_name:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    return "configs/sam2.1/sam2.1_hiera_l.yaml"


def _resolve_sam2_checkpoint(config: Dict) -> str:
    candidates = [
        config.get("SAM2_CHECKPOINT_PATH"),
        config.get("SAM_CHECKPOINT_PATH"),
        config.get("sam2_checkpoint"),
        config.get("sam_checkpoint"),
        "checkpoints/SAM 2.1 Hiera Large.pt",
        "checkpoints/sam2.1_hiera_large.pt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return str(next((c for c in candidates if c), "checkpoints/SAM 2.1 Hiera Large.pt"))


def _mask_records_to_output(mask_records: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
    if not mask_records:
        return {
            "masks": torch.zeros((0, 1, 1), dtype=torch.float32, device=device),
            "iou_preds": torch.zeros((0,), dtype=torch.float32, device=device),
        }

    masks: List[torch.Tensor] = []
    scores: List[float] = []
    for record in mask_records:
        segmentation = record.get("segmentation", record.get("mask"))
        if segmentation is None:
            continue
        if isinstance(segmentation, torch.Tensor):
            mask_tensor = segmentation.float()
        else:
            mask_tensor = torch.from_numpy(np.asarray(segmentation)).float()
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)
        masks.append(mask_tensor.to(device=device))
        scores.append(float(record.get("predicted_iou", record.get("score", 0.0))))

    if not masks:
        return {
            "masks": torch.zeros((0, 1, 1), dtype=torch.float32, device=device),
            "iou_preds": torch.zeros((0,), dtype=torch.float32, device=device),
        }

    return {
        "masks": torch.stack(masks, dim=0),
        "iou_preds": torch.tensor(scores, dtype=torch.float32, device=device),
    }


class GroundingDinoSam2MaskGenerator:
    def __init__(self, config: Dict, device: str):
        if not SAM2_AVAILABLE:
            raise RuntimeError("sam2 package is not installed")

        self.device = torch.device(device)
        self.config = config
        self.sam2_checkpoint = _resolve_sam2_checkpoint(config)
        self.sam2_config = _resolve_sam2_config(self.sam2_checkpoint, config.get("SAM2_CONFIG_FILE"))
        self.predictor = SAM2ImagePredictor(build_sam2(self.sam2_config, self.sam2_checkpoint, device=str(self.device)))
        self.auto_generator = SAM2AutomaticMaskGenerator(
            self.predictor.model,
            points_per_side=config["SAM_NUM_POINTS_PER_SIDE"],
            points_per_batch=config["SAM_NUM_POINTS_PER_BATCH"],
            pred_iou_thresh=config["SAM_PRED_IOU_THRESHOLD"],
            stability_score_thresh=config.get("SAM_STABILITY_SCORE_THRESHOLD", 0.95),
        )

        self.use_grounding_dino = bool(config.get("use_grounding_dino", False)) and GROUNDING_DINO_AVAILABLE
        self.grounding_model = None
        self.grounding_processor = None
        self.grounding_device = torch.device("cpu")

        if self.use_grounding_dino:
            grounding_checkpoint = config.get("GROUNDING_DINO_CHECKPOINT")
            model_id = config.get("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
            try:
                # Prefer a local GroundingDINO checkpoint/folder if provided and usable.
                if grounding_checkpoint and Path(grounding_checkpoint).exists():
                    try:
                        self.grounding_processor = AutoProcessor.from_pretrained(grounding_checkpoint)
                        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_checkpoint)
                    except Exception:
                        # Fallback to HF model id if local checkpoint isn't in HF format
                        self.grounding_processor = AutoProcessor.from_pretrained(model_id)
                        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                else:
                    self.grounding_processor = AutoProcessor.from_pretrained(model_id)
                    self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self.grounding_model = self.grounding_model.to(self.grounding_device).eval()
            except Exception as exc:
                print(f"⚠ GroundingDINO load failed, falling back to SAM2-only: {exc}")
                self.use_grounding_dino = False
                self.grounding_model = None
                self.grounding_processor = None

    def _grounding_boxes(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        if not self.use_grounding_dino or self.grounding_model is None or self.grounding_processor is None:
            return []

        image_pil = Image.fromarray(image.astype(np.uint8))
        prompts = self.config.get(
            "GROUNDING_DINO_PROMPTS",
            "person . dog . cat . chair . table . car . building . tree . plant . road . street . sidewalk . pavement . vegetation . sign . pole . fence . bicycle . motorcycle . bus . truck . sky",
        )

        inputs = self.grounding_processor(images=image_pil, text=prompts, return_tensors="pt")
        inputs = {k: v.to(self.grounding_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=float(self.config.get("GROUNDING_DINO_BOX_THRESHOLD", 0.3)),
            text_threshold=float(self.config.get("GROUNDING_DINO_TEXT_THRESHOLD", 0.25)),
            target_sizes=[image.shape[:2]],
        )

        boxes: List[Tuple[np.ndarray, str]] = []
        if not results:
            return boxes

        det_boxes = results[0].get("boxes", [])
        det_labels = results[0].get("labels", [])
        for det_box, det_label in zip(det_boxes, det_labels):
            box_np = det_box.detach().cpu().numpy().astype(np.float32)
            label = str(det_label).strip() or "unknown"
            boxes.append((box_np, label))
        return boxes

    def _predict_with_box(self, image: np.ndarray, box: np.ndarray) -> List[Dict]:
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(box=box[None, :], multimask_output=True)
        records: List[Dict] = []
        for mask, score in zip(masks, scores):
            records.append({"segmentation": mask, "predicted_iou": float(score)})
        return records

    def _predict_with_points(self, image: np.ndarray, positive_points: np.ndarray, negative_points: Optional[np.ndarray]) -> List[Dict]:
        self.predictor.set_image(image)
        point_coords = positive_points
        point_labels = np.ones((len(positive_points),), dtype=np.int32)
        if negative_points is not None and len(negative_points) > 0:
            point_coords = np.concatenate([positive_points, negative_points], axis=0)
            point_labels = np.concatenate([
                np.ones((len(positive_points),), dtype=np.int32),
                np.zeros((len(negative_points),), dtype=np.int32),
            ], axis=0)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        return [{"segmentation": mask, "predicted_iou": float(score)} for mask, score in zip(masks, scores)]

    def generate(self, image: np.ndarray, positive_points=None, negative_points=None) -> Dict[str, torch.Tensor]:
        records: List[Dict] = []

        # GroundingDINO -> SAM2 box prompts, if enabled.
        if self.use_grounding_dino:
            boxes = self._grounding_boxes(image)
            if positive_points is not None and len(positive_points) > 0:
                h, w = image.shape[:2]
                pts = np.asarray(positive_points, dtype=np.float32)
                pts_px = np.stack([pts[:, 0] * float(w), pts[:, 1] * float(h)], axis=1)
            else:
                pts_px = None

            for box, _label in boxes:
                if pts_px is not None and not np.any(
                    (pts_px[:, 0] >= box[0]) & (pts_px[:, 0] <= box[2]) &
                    (pts_px[:, 1] >= box[1]) & (pts_px[:, 1] <= box[3])
                ):
                    continue
                records.extend(self._predict_with_box(image, box))

        # Fallbacks for point-prompted or fully automatic segmentation.
        if not records:
            if positive_points is not None and len(positive_points) > 0:
                h, w = image.shape[:2]
                pts = np.asarray(positive_points, dtype=np.float32)
                if pts.max() <= 1.5:
                    pts[:, 0] *= float(w)
                    pts[:, 1] *= float(h)
                neg_pts = None
                if negative_points is not None and len(negative_points) > 0:
                    neg_pts = np.asarray(negative_points, dtype=np.float32)
                    if neg_pts.max() <= 1.5:
                        neg_pts[:, 0] *= float(w)
                        neg_pts[:, 1] *= float(h)
                records = self._predict_with_points(image, pts, neg_pts)
            else:
                records = self.auto_generator.generate(image)

        return _mask_records_to_output(records, self.device)


def get_sam_model(config: Dict, device: str):
    variant = str(config["sam_variant"]).lower()
    if variant == "mobile":
        mobile_checkpoint = config["MOBILE_SAM_CHECKPOINT_PATH"]
        checkpoint = torch.load(mobile_checkpoint)
        mobile_sam = setup_mobile_sam()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        return SamAutomaticMaskGenerator(
            mobile_sam,
            points_per_side=config["SAM_NUM_POINTS_PER_SIDE"],
            points_per_batch=config["SAM_NUM_POINTS_PER_BATCH"],
            pred_iou_thresh=config["SAM_PRED_IOU_THRESHOLD"],
        )

    if variant == "original":
        sam = sam_model_registry[config["SAM_ENCODER_VERSION"]](checkpoint=config["SAM_CHECKPOINT_PATH"]).to(
            device=device
        )
        return SamAutomaticMaskGenerator(
            sam,
            points_per_side=config["SAM_NUM_POINTS_PER_SIDE"],
            points_per_batch=config["SAM_NUM_POINTS_PER_BATCH"],
            pred_iou_thresh=config["SAM_PRED_IOU_THRESHOLD"],
        )

    if variant == "sam2":
        return GroundingDinoSam2MaskGenerator(config, device)

    raise ValueError(f"Unknown SAM variant: {config.get('sam_variant')!r}")


def auto_segment(
    config: Dict,
    auto_sam,
    image: np.ndarray,
    forward_mask: Optional[torch.Tensor],
    min_side: int,
    suppress_small_mask: bool,
) -> Tuple[torch.Tensor, List[ObjectInfo]]:
    """Segment a single frame using the configured mask generator."""
    device = getattr(auto_sam, "device", getattr(getattr(auto_sam, "predictor", None), "device", torch.device("cpu")))

    h, w = image.shape[:2]
    if min_side > 0:
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    if forward_mask is not None:
        foreground_mask = (forward_mask > 0).float().unsqueeze(0).unsqueeze(0)
        foreground_mask = F.interpolate(
            foreground_mask,
            scale_factor=1 / 16,
            mode="bilinear",
            antialias=True,
        )
        n_per_side = config["SAM_NUM_POINTS_PER_SIDE"]
        offset = 1 / (2 * n_per_side)
        points_one_side = torch.linspace(offset, 1 - offset, n_per_side, device=device)
        points_x = points_one_side.unsqueeze(0).repeat(n_per_side, 1)
        points_y = points_one_side.unsqueeze(1).repeat(1, n_per_side)
        points = torch.stack([points_x, points_y], dim=-1).unsqueeze(0)
        points_label = F.grid_sample(foreground_mask, points * 2 - 1, align_corners=False).view(-1)
        points = points.view(-1, 2)
        positive_points = points[points_label < 0.01].cpu().numpy()
        if len(positive_points) == 0:
            output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
            return output_mask, []
        negative_points = None
        mask_data = auto_sam.generate(image, positive_points, negative_points)
    else:
        mask_data = auto_sam.generate(image)

    curr_id = 1
    segments_info: List[ObjectInfo] = []
    pred_masks = mask_data["masks"].float()
    predicted_iou = mask_data["iou_preds"]

    if pred_masks.shape[0] == 0:
        output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
        return output_mask, segments_info

    pred_masks = F.interpolate(pred_masks.unsqueeze(0), (new_h, new_w), mode="bilinear")[0]

    if suppress_small_mask:
        areas = pred_masks.flatten(-2).sum(-1)
        scores = areas.unsqueeze(-1).unsqueeze(-1)
        scored_masks = pred_masks * scores
        scored_masks_with_bg = torch.cat(
            [torch.zeros((1, *pred_masks.shape[1:]), device=device) + 0.1, scored_masks], dim=0
        )
        output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)

        hard_mask = torch.argmax(scored_masks_with_bg, dim=0)
        for k in range(scores.shape[0]):
            mask_area = (hard_mask == (k + 1)).sum()
            original_area = (pred_masks[k] > 0.5).sum()
            mask = (hard_mask == (k + 1)) & (pred_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum() > 0:
                if mask_area / original_area < config["SAM_OVERLAP_THRESHOLD"]:
                    continue
                output_mask[mask] = curr_id
                segments_info.append(ObjectInfo(id=curr_id, score=predicted_iou[k].item()))
                curr_id += 1
    else:
        areas = pred_masks.flatten(-2).sum(-1)
        scores = (areas.max() * 2 - areas).unsqueeze(-1).unsqueeze(-1)
        scored_masks = pred_masks * scores
        scored_masks_with_bg = torch.cat(
            [torch.zeros((1, *scored_masks.shape[1:]), device=device) + 0.1, scored_masks], dim=0
        )
        output_mask = torch.argmax(scored_masks_with_bg, dim=0)
        for k in range(scored_masks.shape[0]):
            mask = output_mask == (k + 1)
            if mask.sum() > 0:
                segments_info.append(ObjectInfo(id=curr_id, score=predicted_iou[k].item()))
                curr_id += 1

    return output_mask, segments_info
