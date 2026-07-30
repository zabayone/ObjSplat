# Local six-scene evaluation dataset

Place six 2:1 equirectangular panoramas in this directory:

```text
indoor_1.jpg
indoor_2.jpg
indoor_3.jpg
outdoor_1.jpg
outdoor_2.jpg
outdoor_3.jpg
```

JPEG and PNG inputs are supported; if an extension changes, update
`benchmark/configs/evaluation_six_scenes.yaml` and
`benchmark/configs/evaluation_focused_ablations.yaml`.

The benchmark validates the 2:1 aspect ratio, applies EXIF orientation, converts
each source to a true RGB PNG, and creates a normalized `4096×2048` working copy
under `benchmark_work/`. Original files are never modified.

Panoramas are intentionally excluded from Git because of their size and possible
dataset licensing restrictions. Do not commit images unless redistribution
rights are confirmed.

Run the complete 22-run suite from the repository root:

```bash
bash benchmark/run_evaluation_suite.sh
```
