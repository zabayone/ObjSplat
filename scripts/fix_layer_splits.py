#!/usr/bin/env python3
"""Fix layer splits/merges by using GroundingDINO tag boxes and small-component reassign.

Produces fixed masks under `outputs_lgs/traindata_fixed/` and a visualization overlay.
"""
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import sys

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    HF_OK = True
except Exception:
    HF_OK = False

from scipy import ndimage
import torch


def load_layer_masks(traindata_path: Path):
    masks = {}
    for d in sorted(traindata_path.iterdir()):
        if d.is_dir() and d.name.startswith('layer'):
            mf = d / f"{d.name}_erp_mask.png"
            if mf.exists():
                masks[d.name] = np.array(Image.open(mf).convert('L')) > 127
    return masks


def run_grounding(pano_path: Path, prompts: str):
    if not HF_OK:
        raise RuntimeError('transformers or grounding-dino not available in environment')
    proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
    model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
    model = model.to('cpu').eval()
    img = Image.open(pano_path).convert('RGB')
    inputs = proc(images=img, text=prompts, return_tensors='pt')
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with np.errstate():
        with torch.no_grad():
            outputs = model(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, inputs['input_ids'], box_threshold=0.25, text_threshold=0.2, target_sizes=[img.size[::-1]]
    )
    if not results:
        return []
    boxes = results[0].get('boxes', [])
    labels = results[0].get('labels', [])
    tag_boxes = []
    for b, l in zip(boxes, labels):
        tag_boxes.append((np.array(b.tolist(), dtype=int), str(l)))
    return tag_boxes


def box_to_mask(box, H, W):
    x1, y1, x2, y2 = box
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def save_masks(out_dir: Path, masks: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        im = Image.fromarray((mask * 255).astype('uint8'))
        im.save(out_dir / f"{name}_erp_mask.png")


def visualize_overlay(out_path: Path, pano_path: Path, masks: dict):
    pano = np.array(Image.open(pano_path).convert('RGB'))
    H, W = pano.shape[:2]
    overlay = pano.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 64, 0), (0,128,0)]
    i = 0
    for name, mask in masks.items():
        c = colors[i % len(colors)]
        alpha = 0.4
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(c) * alpha).astype(np.uint8)
        i += 1
    Image.fromarray(overlay).save(out_path)


def main():
    pano = Path('outputs_lgs/rgb.png')
    traind = Path('outputs_lgs/traindata')
    out = Path('outputs_lgs/traindata_fixed')
    out.mkdir(exist_ok=True)

    masks = load_layer_masks(traind)
    if not masks:
        print('No layer masks found'); sys.exit(1)
    names = list(masks.keys())
    H, W = next(iter(masks.values())).shape
    pano_area = H * W

    # run grounding and build tag masks
    prompts = (
        'person . tree . sky . sidewalk pavement . grass . road . building . bush . bench . bicycle . car'
    )
    try:
        tag_boxes = run_grounding(pano, prompts)
    except Exception as e:
        print('Grounding failed:', e)
        tag_boxes = []

    tag_masks = defaultdict(lambda: np.zeros((H, W), dtype=bool))
    for box, tag in tag_boxes:
        m = box_to_mask(box, H, W)
        if m is None:
            continue
        tag_masks[tag] |= m

    # create new masks dict for output
    new_masks = {}
    used = np.zeros((H, W), dtype=bool)

    # split large layers by tags
    large_threshold = 0.05 * pano_area
    for lname, lm in masks.items():
        area = lm.sum()
        if area >= large_threshold:
            # cut out tag regions first
            remaining = lm.copy()
            for tag, tmask in tag_masks.items():
                inter = remaining & tmask
                if inter.sum() > 0:
                    new_name = f"{lname}__{tag}"
                    new_masks[new_name] = inter
                    remaining = remaining & (~tmask)
            # leftover part
            if remaining.sum() > 0:
                new_masks[lname] = remaining
            used |= lm
        else:
            new_masks[lname] = lm
            used |= lm

    # reassign small components: if component < 0.5% pano, assign to neighbor layer with max overlap
    threshold = int(0.005 * pano_area)
    for name, mask in list(new_masks.items()):
        labeled, num = ndimage.label(mask)
        if num == 0:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        for comp_idx, size in enumerate(sizes, start=1):
            if size >= threshold:
                continue
            comp_mask = labeled == comp_idx
            # find overlap with other masks
            best = None
            best_overlap = 0
            for other_name, other_mask in new_masks.items():
                if other_name == name:
                    continue
                inter = (comp_mask & other_mask).sum()
                if inter > best_overlap:
                    best_overlap = inter
                    best = other_name
            if best and best_overlap > 0:
                # move comp to best
                new_masks[best] = new_masks[best] | comp_mask
                new_masks[name] = new_masks[name] & (~comp_mask)

    # cleanup: remove empty masks, save
    final_masks = {n: m for n, m in new_masks.items() if m.sum() > 0}
    save_masks(out, final_masks)
    visualize_overlay(out / 'layer_mask_visualization_fixed.png', pano, final_masks)
    print('Saved fixed masks to', out)


if __name__ == '__main__':
    main()
