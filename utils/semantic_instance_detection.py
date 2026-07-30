#!/usr/bin/env python3
"""Semantic instance detection for object identification in LayerPano3D.

Uses Segment Anything (SAM) + optional GroundingDINO for semantic-aware
instance detection, identifying actual objects rather than just regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import time

import numpy as np
import torch
import cv2
from PIL import Image
from benchmark.runtime_hooks import record_stage

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# GroundingDINO for semantic tagging
GROUNDING_DINO_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    pass

SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except Exception:
    build_sam2 = None  # type: ignore[assignment]
    SAM2ImagePredictor = None  # type: ignore[assignment]


def _label_tokens(label: Optional[str]) -> set[str]:
    raw = str(label or "").strip().lower()
    if not raw:
        return set()
    return {part for part in raw.replace("/", " ").replace("-", " ").replace("_", " ").split() if part}


def _is_vegetation_label(label: Optional[str]) -> bool:
    tokens = _label_tokens(label)
    vegetation_tokens = {
        "tree",
        "trees",
        "plant",
        "plants",
        "bush",
        "bushes",
        "shrub",
        "shrubs",
        "grass",
        "vegetation",
        "foliage",
        "forest",
        "wood",
        "woods",
        "leaf",
        "leaves",
        "branch",
        "branches",
    }
    return bool(tokens & vegetation_tokens)


class SemanticInstanceDetector:
    """Detect and classify objects in images using SAM + optional GroundingDINO."""
    
    def __init__(self, checkpoint_path: str = "checkpoints/sam_vit_h_4b8939.pth", 
                 device: str = "mps", 
                 use_grounding: bool = True,
                 grounding_checkpoint: str = "IDEA-Research/grounding-dino-base"):
        """Initialize SAM model for instance detection.
        
        Args:
            checkpoint_path: Path to SAM checkpoint
            device: Device for inference ('mps', 'cuda', 'cpu')
            use_grounding: Enable semantic tagging with GroundingDINO
            grounding_checkpoint: Path to GroundingDINO checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.sam_model = None
        self.mask_generator = None
        
        self.use_grounding = use_grounding and GROUNDING_DINO_AVAILABLE
        self.grounding_model = None
        self.grounding_processor = None
        self.grounding_checkpoint = grounding_checkpoint
        self.grounding_device = torch.device("cpu")
        
        if not SAM_AVAILABLE:
            print("[WARN] SAM not available, semantic instance detection unavailable")
            return
        
        if not Path(checkpoint_path).exists():
            print(f"[WARN] SAM checkpoint not found at {checkpoint_path}")
            return
        
        self._load_model()
        if self.use_grounding:
            self._load_grounding_model()
    
    def _load_model(self) -> None:
        """Load SAM model."""
        try:
            self.sam_model = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
            self.sam_model = self.sam_model.to(self.device).eval()
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=40,
                pred_iou_thresh=0.78,
                stability_score_thresh=0.85,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=50,
            )
            print(f"[OK] SAM model loaded from {self.checkpoint_path}")
        except Exception as e:
            print(f"[WARN] Failed to load SAM: {e}")
            self.sam_model = None
            self.mask_generator = None
    
    def _load_grounding_model(self) -> None:
        """Load GroundingDINO model for semantic tagging."""
        try:
            if not GROUNDING_DINO_AVAILABLE:
                print("[WARN] GroundingDINO not available")
                self.use_grounding = False
                return
            model_id = "IDEA-Research/grounding-dino-base"
            grounding_checkpoint = self.grounding_checkpoint
            print("  Loading GroundingDINO (local checkpoint preferred if present)...")
            try:
                if grounding_checkpoint and Path(grounding_checkpoint).exists():
                    # Try loading from local checkpoint/folder first
                    self.grounding_processor = AutoProcessor.from_pretrained(grounding_checkpoint)
                    self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_checkpoint)
                else:
                    self.grounding_processor = AutoProcessor.from_pretrained(model_id)
                    self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            except Exception:
                # Fallback to HF model id
                self.grounding_processor = AutoProcessor.from_pretrained(model_id)
                self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

            self.grounding_model = self.grounding_model.to(self.grounding_device).eval()
            print(f"[OK] GroundingDINO loaded (local:{bool(grounding_checkpoint and Path(grounding_checkpoint).exists())}) on {self.grounding_device}")
        except Exception as e:
            print(f"[WARN] Failed to load GroundingDINO: {e}")
            self.use_grounding = False
            self.grounding_model = None
            self.grounding_processor = None
    
    def _tag_with_grounding(self, image: np.ndarray, masks: List[Dict]) -> Dict[int, str]:
        """Tag SAM masks with semantic labels using GroundingDINO.
        
        Args:
            image: Input image (H, W, 3) uint8
            masks: List of SAM mask dicts
        
        Returns:
            Dict mapping instance_id to semantic tag (e.g. "person", "chair")
        """
        if not self.use_grounding or self.grounding_model is None or self.grounding_processor is None:
            return {}
        
        try:
            image_pil = Image.fromarray(image.astype(np.uint8))
            tags: Dict[int, str] = {}

            # Generic prompts for common object classes
            prompts = (
                "person . dog . cat . chair . table . car . building . tree . plant . road . street . sidewalk . pavement . vegetation "
                ". sign . pole . fence . bicycle . motorcycle . bus . truck . sky . grass . bush"
            )

            # Downscale for faster/stabler GroundingDINO inference on large ERP panoramas.
            max_side = 1024
            h, w = image.shape[:2]
            scale = min(1.0, float(max_side) / float(max(h, w)))
            if scale < 1.0:
                infer_w = max(1, int(round(w * scale)))
                infer_h = max(1, int(round(h * scale)))
                infer_pil = image_pil.resize((infer_w, infer_h), Image.BILINEAR)
            else:
                infer_w, infer_h = w, h
                infer_pil = image_pil

            # Run GroundingDINO once per image, then match detections to SAM masks.
            inputs = self.grounding_processor(images=infer_pil, text=prompts, return_tensors="pt")
            inputs = {k: v.to(self.grounding_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=0.30,
                text_threshold=0.25,
                target_sizes=[(infer_h, infer_w)],
            )

            det_boxes = results[0].get("boxes", []) if results else []
            det_labels = results[0].get("labels", []) if results else []

            # Rescale detection boxes back to original image coordinates.
            if scale < 1.0 and len(det_boxes) > 0:
                inv_scale = 1.0 / scale
                det_boxes = [box * inv_scale for box in det_boxes]

            def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = area_a + area_b - inter
                return float(inter / union) if union > 0 else 0.0

            for instance_id, mask_data in enumerate(masks, start=1):
                mask = mask_data["segmentation"]
                ys, xs = np.where(mask)
                if ys.size == 0 or xs.size == 0:
                    tags[instance_id] = "unknown"
                    continue

                sam_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

                best_iou = 0.0
                best_label = "unknown"
                for det_box, det_label in zip(det_boxes, det_labels):
                    box_np = det_box.detach().cpu().numpy().astype(np.float32)
                    iou = _bbox_iou(sam_box, box_np)
                    if iou > best_iou:
                        best_iou = iou
                        best_label = str(det_label).strip() if str(det_label).strip() else "unknown"

                tags[instance_id] = best_label if best_iou > 0.05 else "unknown"

            return tags
        except Exception as e:
            print(f"[WARN] GroundingDINO tagging failed: {e}")
            return {}
    
    def detect_instances(self, image: np.ndarray) -> Dict[int, Dict]:
        """Detect instances in image using SAM automatic masking.
        
        Args:
            image: Input image (H, W, 3) in RGB format, uint8
        
        Returns:
            Dict mapping instance_id to {mask, area, bbox, confidence, tag (optional)}
        """
        if self.mask_generator is None:
            print("[WARN] SAM model not available")
            return {}
        
        try:
            masks = self.mask_generator.generate(image)
            
            # Get semantic tags if GroundingDINO available
            tags = {}
            if self.use_grounding:
                tags = self._tag_with_grounding(image, masks)
            
            instances = {}
            for instance_id, mask_data in enumerate(masks, start=1):
                mask = mask_data["segmentation"]
                area = int(mask.sum())
                
                # Get bounding box
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    bbox = [x1, y1, x2, y2]
                else:
                    bbox = [0, 0, 0, 0]
                
                confidence = float(mask_data.get("predicted_iou", 0.0))
                
                instances[instance_id] = {
                    "mask": mask.astype(np.uint8),
                    "area": area,
                    "bbox": bbox,
                    "confidence": confidence,
                    "tag": tags.get(instance_id, "unknown") if tags else "unknown",
                }
            
            return instances
        except Exception as e:
            print(f"[WARN] Error during instance detection: {e}")
            return {}


def detect_objects_in_layer(
    rgb_image: np.ndarray,
    mask_image: np.ndarray,
    checkpoint_path: str = "checkpoints/sam_vit_h_4b8939.pth",
    device: str = "mps",
    min_area: int = 100,
    use_grounding: bool = True,
    grounding_checkpoint: str = "IDEA-Research/grounding-dino-base",
) -> Tuple[np.ndarray, Optional[Dict[int, str]]]:
    """Detect semantic objects in a layer using SAM + optional GroundingDINO.
    
    Creates a detailed instance map where each connected region of the layer
    is segmented into potential objects using SAM's automatic masking.
    Optionally tags each object with semantic label using GroundingDINO.
    
    Args:
        rgb_image: Panoramic RGB image (H, W, 3)
        mask_image: Layer mask (H, W) binary
        checkpoint_path: Path to SAM checkpoint
        device: Device for inference
        min_area: Minimum object area in pixels
        use_grounding: Enable semantic tagging with GroundingDINO
        grounding_checkpoint: Path to GroundingDINO checkpoint
    
    Returns:
        (instance_map, tags_dict) where:
            instance_map: (H, W, int32) with object IDs
            tags_dict: {instance_id: semantic_tag_string} or None if no GroundingDINO
    """
    detector = SemanticInstanceDetector(
        checkpoint_path, device, 
        use_grounding=use_grounding, 
        grounding_checkpoint=grounding_checkpoint
    )
    
    if detector.sam_model is None:
        print("[WARN] Cannot detect objects without SAM")
        return np.zeros_like(mask_image, dtype=np.int32), None
    
    # Detect instances on full RGB to avoid suppressing boundaries,
    # then intersect with the layer mask.
    h, w = rgb_image.shape[:2]
    instances = detector.detect_instances(rgb_image)
    
    # Sort instances by area (descending) so larger objects are assigned first
    # and smaller objects fill remaining pixels (avoids overwriting large objects)
    sorted_instances = sorted(instances.items(), key=lambda x: x[1]["area"], reverse=True)
    
    # Build instance map without overwriting already-assigned pixels
    instance_map = np.zeros((h, w), dtype=np.int32)
    tags_dict = {}
    
    for instance_id, data in sorted_instances:
        inst_mask = data["mask"]
        area = data["area"]
        
        # Only keep objects within the layer mask and above min area
        if area >= min_area:
            valid_mask = inst_mask & mask_image
            if valid_mask.sum() > min_area:
                # Only assign pixels that aren't already assigned
                # This prevents smaller objects from overwriting larger ones
                unassigned_pixels = valid_mask & (instance_map == 0)
                instance_map[unassigned_pixels] = instance_id
                
                # Store semantic tag if available
                if "tag" in data:
                    tags_dict[instance_id] = data["tag"]
    
    return instance_map, tags_dict if tags_dict else None


def detect_all_objects_in_panorama(
    rgb_path: Path,
    checkpoint_path: str = "checkpoints/sam_vit_h_4b8939.pth",
    device: str = "mps",
    use_grounding: bool = True,
) -> Dict:
    """Detect all salient objects in panoramic image.
    
    Uses SAM to identify all prominent objects regardless of layer,
    optionally tags with semantic labels using GroundingDINO.
    
    Args:
        rgb_path: Path to panoramic RGB image
        checkpoint_path: Path to SAM checkpoint
        device: Device for inference
        use_grounding: Enable semantic tagging
    
    Returns:
        Dict with 'instance_map', 'instances_count', 'instances', and 'tags' (if available)
    """
    if not rgb_path.exists():
        print(f"[WARN] RGB image not found: {rgb_path}")
        return {}
    
    detector = SemanticInstanceDetector(checkpoint_path, device, use_grounding=use_grounding)
    if detector.sam_model is None:
        return {}
    
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    instances = detector.detect_instances(rgb)
    
    # Build instance map
    h, w = rgb.shape[:2]
    instance_map = np.zeros((h, w), dtype=np.int32)
    tags = {}
    
    for instance_id, data in instances.items():
        mask = data["mask"]
        instance_map[mask] = instance_id
        if "tag" in data:
            tags[instance_id] = data["tag"]
    
    result = {
        "instance_map": instance_map,
        "instances_count": len(instances),
        "instances": instances
    }
    
    if tags:
        result["tags"] = tags
    
    return result


def detect_objects_grounding_then_sam_on_panorama(
    rgb_path: Path,
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth",
    device: str = "mps",
    use_grounding: bool = True,
    grounding_checkpoint: str = "IDEA-Research/grounding-dino-base",
    sam_variant: str = "original",
    grounding_prompts: Optional[str] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    max_detections: Optional[int] = None,
    multimask_output: bool = True,
    min_mask_area: int = 1500,
    sam2_config: Optional[str] = None,
    box_padding_ratio: float = 0.12,
    grounding_infer_max_side: int = 1024,
    box_nms_threshold: float = 0.75,
    min_component_area_ratio: float = 0.01,
    morph_open_kernel: int = 3,
) -> Dict[str, Any]:
    """Run GroundingDINO to get boxes, then run SAM predictor on those boxes.

    Returns a dict with:
        - instance_map: (H,W) int32 map of instance ids
        - masks: {id: boolean mask}
        - tags: {id: label}
    """
    rgb_path = Path(rgb_path)
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB not found: {rgb_path}")

    pil = Image.open(rgb_path).convert("RGB")
    img = np.array(pil, dtype=np.uint8)
    h, w = img.shape[:2]

    # Load GroundingDINO if requested
    proc = None
    gmodel = None
    det_boxes: List[Tuple[np.ndarray, str, float]] = []
    grounding_error: Optional[str] = None
    grounding_succeeded = False
    grounding_started = time.perf_counter()
    if use_grounding and GROUNDING_DINO_AVAILABLE:
        try:
            model_id = "IDEA-Research/grounding-dino-base"
            if grounding_checkpoint and Path(grounding_checkpoint).exists():
                try:
                    proc = AutoProcessor.from_pretrained(grounding_checkpoint)
                    gmodel = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_checkpoint)
                except Exception:
                    proc = AutoProcessor.from_pretrained(model_id)
                    gmodel = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            else:
                proc = AutoProcessor.from_pretrained(model_id)
                gmodel = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            gmodel = gmodel.to("cpu").eval()

            prompts = grounding_prompts or (
                "person . dog . cat . chair . table . car . building . tree . plant . road . street . sidewalk . pavement . vegetation . sign . pole . fence . bicycle . motorcycle . bus . truck . bench . sofa . bed . cabinet . sky . grass"
            )
            max_side = max(256, int(grounding_infer_max_side))
            scale = min(1.0, float(max_side) / float(max(h, w)))
            if scale < 1.0:
                infer_w = max(1, int(round(w * scale)))
                infer_h = max(1, int(round(h * scale)))
                infer_pil = pil.resize((infer_w, infer_h), Image.BILINEAR)
            else:
                infer_pil = pil

            inputs = proc(images=infer_pil, text=prompts, return_tensors="pt")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = gmodel(**inputs)
            results = proc.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=float(box_threshold),
                text_threshold=float(text_threshold),
                target_sizes=[(infer_pil.size[1], infer_pil.size[0])],
            )
            if results:
                boxes = results[0].get("boxes", [])
                labels = results[0].get("labels", [])
                scores = results[0].get("scores", [])
                if scale < 1.0 and len(boxes) > 0:
                    inv_scale = 1.0 / scale
                    boxes = [box * inv_scale for box in boxes]
                for idx, (b, l) in enumerate(zip(boxes, labels)):
                    box_np = b.detach().cpu().numpy() if hasattr(b, 'detach') else np.array(b)
                    x1, y1, x2, y2 = [float(x) for x in box_np.tolist()]
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(0, min(w, x2))
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    score = float(scores[idx].detach().cpu().item()) if idx < len(scores) and hasattr(scores[idx], "detach") else 0.0
                    det_boxes.append((np.array([x1, y1, x2, y2], dtype=np.float32), str(l).strip() or "unknown", score))
            grounding_succeeded = bool(det_boxes)
        except Exception as e:
            print(f"[WARN] GroundingDINO failed: {e}")
            grounding_error = str(e)
    elif use_grounding:
        grounding_error = "transformers GroundingDINO support is unavailable"
    record_stage(
        "object_detection",
        time.perf_counter() - grounding_started,
        status="success" if grounding_succeeded or not use_grounding else "failed",
    )

    if not det_boxes:
        det_boxes = [(np.array([0, 0, w, h], dtype=np.float32), "panorama", 0.0)]

    det_boxes = sorted(det_boxes, key=lambda item: item[2], reverse=True)
    if max_detections is not None and int(max_detections) > 0:
        det_boxes = det_boxes[: int(max_detections)]

    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a.astype(np.float32).tolist()
        bx1, by1, bx2, by2 = box_b.astype(np.float32).tolist()
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    if box_nms_threshold and float(box_nms_threshold) > 0 and det_boxes:
        kept: List[Tuple[np.ndarray, str, float]] = []
        for box, label, score in det_boxes:
            label_norm = str(label).strip().lower()
            duplicate = any(
                str(kept_label).strip().lower() == label_norm
                and _bbox_iou(box, kept_box) >= float(box_nms_threshold)
                for kept_box, kept_label, _kept_score in kept
            )
            if not duplicate:
                kept.append((box, label, score))
        det_boxes = kept

    def _expand_box(box: np.ndarray, image_w: int, image_h: int, padding_ratio: float) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(np.float32).tolist()
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * float(padding_ratio)
        pad_y = bh * float(padding_ratio)
        return np.array(
            [
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(image_w), x2 + pad_x),
                min(float(image_h), y2 + pad_y),
            ],
            dtype=np.float32,
        )

    def _mask_inside_box(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        clipped = np.zeros((h, w), dtype=bool)
        if x2 <= x1 or y2 <= y1:
            return clipped
        clipped[y1:y2, x1:x2] = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
        return clipped

    def _component_overlaps_box(labels: np.ndarray, comp_idx: int, box: np.ndarray) -> bool:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return False
        return bool(np.any(labels[y1:y2, x1:x2] == comp_idx))

    def _keep_significant_component(mask: np.ndarray, anchor_box: np.ndarray) -> np.ndarray:
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        if int(mask_u8.sum()) == 0:
            return mask.astype(bool)
        if int(morph_open_kernel) > 1:
            k = int(morph_open_kernel)
            if k % 2 == 0:
                k += 1
            kernel = np.ones((k, k), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            if int(mask_u8.sum()) == 0:
                return np.asarray(mask, dtype=bool)
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n_labels <= 2:
            return mask_u8.astype(bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return mask_u8.astype(bool)
        max_area = int(areas.max())
        total_area = int(areas.sum())
        min_component_area = max(
            16,
            int(max_area * 0.05),
            int(total_area * float(min_component_area_ratio)),
        )
        best_idx = None
        best_area = -1
        for comp_idx, area in enumerate(areas, start=1):
            if int(area) >= min_component_area and _component_overlaps_box(labels, comp_idx, anchor_box):
                if int(area) > best_area:
                    best_area = int(area)
                    best_idx = comp_idx
        if best_idx is None:
            best_idx = int(np.argmax(areas)) + 1
        return labels == best_idx

    def _clean_prompt_mask(mask: np.ndarray, box: np.ndarray, label: str) -> np.ndarray:
        label_norm = str(label).strip().lower()
        label_tokens = _label_tokens(label)
        if "sky" in label_tokens:
            # GroundingDINO boxes are often much tighter than the actual sky,
            # especially on 2:1 ERPs. SAM already predicts the semantic region;
            # clipping it back to the detection box collapses the sky to a
            # narrow strip and destroys gaps between vegetation.
            return np.asarray(mask, dtype=bool)
        if _is_vegetation_label(label):
            # Trees, shrubs, and foliage often fragment into thin crowns and
            # disconnected branches. Preserve their full SAM extent instead of
            # trimming the mask back to the detection box and discarding the
            # outer canopy.
            mask_u8 = np.asarray(mask, dtype=np.uint8)
            if int(mask_u8.sum()) == 0:
                return np.asarray(mask, dtype=bool)
            k = max(3, int(round(min(w, h) / 320.0)))
            if k % 2 == 0:
                k += 1
            kernel = np.ones((k, k), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            expanded_box = _expand_box(box, w, h, max(float(box_padding_ratio), 0.18))
            clipped = _mask_inside_box(mask_u8.astype(bool), expanded_box)
            if int(clipped.sum()) >= max(1, int(mask_u8.sum() * 0.35)):
                return clipped
            return mask_u8.astype(bool)
        expanded_box = _expand_box(box, w, h, box_padding_ratio)
        clipped = _mask_inside_box(mask, expanded_box)
        clipped = _keep_significant_component(clipped, box)
        return _mask_inside_box(clipped, expanded_box)

    def _mask_priority(record: Tuple[np.ndarray, str, float]) -> Tuple[int, int, float]:
        mask, label, score = record
        label_norm = str(label).strip().lower()
        label_tokens = _label_tokens(label_norm)
        stuff_labels = {
            "sky",
            "road",
            "street",
            "sidewalk",
            "pavement",
            "ground",
            "grass",
            "floor",
            "wall",
            "building",
        }
        if "sky" in label_tokens:
            semantic_priority = 1
        elif any(token in label_norm for token in stuff_labels):
            semantic_priority = 2
        else:
            semantic_priority = 0
        area = int(np.asarray(mask, dtype=bool).sum())
        return (semantic_priority, area, -float(score))

    def _resolve_sam2_config(checkpoint_path: str) -> str:
        if sam2_config:
            return sam2_config
        lower_name = Path(checkpoint_path).name.lower()
        if "tiny" in lower_name:
            return "configs/sam2.1/sam2.1_hiera_t.yaml"
        if "small" in lower_name:
            return "configs/sam2.1/sam2.1_hiera_s.yaml"
        if "base" in lower_name or "b+" in lower_name:
            return "configs/sam2.1/sam2.1_hiera_b+.yaml"
        return "configs/sam2.1/sam2.1_hiera_l.yaml"

    def _predict_box_masks() -> List[Tuple[np.ndarray, str, float]]:
        variant = str(sam_variant or "original").lower()
        records: List[Tuple[np.ndarray, str, float]] = []

        if variant == "sam2":
            if not SAM2_AVAILABLE:
                raise RuntimeError("sam2 package is not installed")
            predictor = SAM2ImagePredictor(
                build_sam2(_resolve_sam2_config(sam_checkpoint), sam_checkpoint, device=str(device))
            )
            predictor.set_image(img)
            for box, label, det_score in det_boxes:
                masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=bool(multimask_output))
                best_idx = int(np.argmax(scores)) if len(scores) else 0
                mask = _clean_prompt_mask(masks[best_idx].astype(bool), box, label)
                records.append((mask, label, float(scores[best_idx]) if len(scores) else det_score))
            return records

        if not SAM_AVAILABLE:
            raise RuntimeError("segment-anything package is not installed")
        model_type = "vit_h"
        lower = Path(sam_checkpoint).name.lower()
        if "vit_l" in lower:
            model_type = "vit_l"
        elif "vit_b" in lower:
            model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam = sam.to(device=torch.device(device)).eval()
        predictor = SamPredictor(sam)
        predictor.set_image(img)
        for box, label, det_score in det_boxes:
            masks, scores, _ = predictor.predict(box=box, multimask_output=bool(multimask_output))
            best_idx = int(np.argmax(scores)) if len(scores) else 0
            mask = _clean_prompt_mask(masks[best_idx].astype(bool), box, label)
            records.append((mask, label, float(scores[best_idx]) if len(scores) else det_score))
        return records

    sam_error: Optional[str] = None
    sam_succeeded = False
    segmentation_started = time.perf_counter()
    try:
        mask_records = _predict_box_masks()
        sam_succeeded = bool(mask_records)
        if not sam_succeeded:
            raise RuntimeError("SAM returned no masks")
    except Exception as e:
        print(f"[WARN] SAM box prompting failed, falling back to full-panorama mask: {e}")
        sam_error = str(e)
        mask_records = [(np.ones((h, w), dtype=bool), "panorama-fallback", 0.0)]
    record_stage(
        "segmentation",
        time.perf_counter() - segmentation_started,
        status="success" if sam_succeeded else "failed",
    )

    instance_map = np.zeros((h, w), dtype=np.int32)
    masks: Dict[int, np.ndarray] = {}
    tags: Dict[int, str] = {}
    scores_out: Dict[int, float] = {}

    # Reserve foreground first, then sky, then ground/layout regions. This
    # prevents broad road masks from consuming sky pixels while preserving
    # object silhouettes in front of the sky.
    mask_records = sorted(mask_records, key=_mask_priority)
    next_id = 1
    for mask, label, score in mask_records:
        mask = np.asarray(mask, dtype=bool)
        if int(mask.sum()) < int(min_mask_area):
            continue
        write_mask = mask & (instance_map == 0)
        if int(write_mask.sum()) < int(min_mask_area):
            continue
        instance_map[write_mask] = next_id
        masks[next_id] = write_mask
        tags[next_id] = label
        scores_out[next_id] = float(score)
        next_id += 1

    if not masks:
        instance_map[:, :] = 1
        masks = {1: np.ones((h, w), dtype=bool)}
        tags = {1: "panorama-fallback"}
        scores_out = {1: 0.0}

    return {
        "instance_map": instance_map,
        "masks": masks,
        "tags": tags,
        "scores": scores_out,
        "detections": [
            {"box": box.tolist(), "label": label, "score": float(score)}
            for box, label, score in det_boxes
        ],
        "grounding_status": {
            "requested": bool(use_grounding),
            "available": bool(GROUNDING_DINO_AVAILABLE),
            "succeeded": bool(grounding_succeeded),
            "error": grounding_error,
        },
        "sam_status": {
            "variant": str(sam_variant),
            "succeeded": bool(sam_succeeded),
            "error": sam_error,
        },
    }
