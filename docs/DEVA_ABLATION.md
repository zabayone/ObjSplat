# DEVA Ablation Runner

This tool runs multiple DEVA mask variants and generates a small HTML report with links to ERP overlays per layer.

## Run

```bash
/opt/anaconda3/envs/layerpano3d/bin/python scripts/deva_ablation.py \
  --input_dir outputs_lgs \
  --output_dir outputs_lgs/ablation
```

Open the report:

- outputs_lgs/ablation/index.html

## Custom variants

Provide a JSON file containing a list of variants. Example structure:

```json
[
  {
    "name": "custom",
    "n_views": 10,
    "phi_bands": [45, 0, -45],
    "min_frame_area": 2000,
    "min_frames": 3,
    "min_total_pixels": 10000,
    "min_points_3d": 5000,
    "sam_pred_iou_threshold": 0.9,
    "sam_stability_score_threshold": 0.93,
    "mask_min_area": 900,
    "detection_every": 4,
    "max_num_objects": 200
  }
]
```

Then run:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python scripts/deva_ablation.py \
  --input_dir outputs_lgs \
  --output_dir outputs_lgs/ablation \
  --variants_json /path/to/variants.json
```
