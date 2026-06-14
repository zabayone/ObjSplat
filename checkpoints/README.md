# Checkpoints

Model weights are intentionally not committed.

Expected local files for the default Grounding-SAM pipeline:

- `SAM 2.1 Hiera Large.pt`
- `sam_vit_h_4b8939.pth`
- `groundingdino_swinb_cogvlm.pth`
- `depth_anything_v2_vitl.pth`

Optional legacy/auxiliary files:

- `DEVA-propagation.pth`
- `ControlNetLama.pth`

Use `scripts/download_checkpoints.py` for direct URLs or Hugging Face files, or place the files in this directory manually.
