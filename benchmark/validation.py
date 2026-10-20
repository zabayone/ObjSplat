"""Validate stored evaluation splits before accepting thesis benchmark scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate_view_split(scene_root: Path, *, evaluation_fraction: float | None = None,
                        seed: int | None = None, require_held_out: bool = True) -> dict:
    root = Path(scene_root) / 'traindata'
    metadata = json.loads((root / 'layer_instances.json').read_text())
    split = metadata.get('benchmark_view_split') or {}
    train = [int(x) for x in split.get('training_indices', [])]
    evaluation = [int(x) for x in split.get('evaluation_indices', [])]
    if not evaluation and not require_held_out:
        return split
    if not train or not evaluation:
        raise ValueError('Missing non-empty training/held-out split; regenerate preprocessing')
    if len(set(train)) != len(train) or len(set(evaluation)) != len(evaluation) or set(train) & set(evaluation):
        raise ValueError('Duplicate or overlapping training/evaluation view IDs')
    frames = root / 'perspective_frames' / 'frames'
    available = sorted(int(p.stem.rsplit('_', 1)[1]) for p in frames.glob('rgb_*.png'))
    if set(available) != set(train) | set(evaluation):
        raise ValueError('Stored split does not partition the perspective frames')
    for index in available:
        if not (frames / f'transform_matrix_{index}.npy').exists():
            raise ValueError(f'Missing camera pose for view {index}')
    for path in root.glob('layer*/frames/rgb_*.png'):
        if int(path.stem.rsplit('_', 1)[1]) in evaluation:
            raise ValueError(f'Held-out view leaked into training: {path}')
    if evaluation_fraction is not None and seed is not None:
        count = min(max(1, round(len(available) * evaluation_fraction)), len(available) - 1)
        expected = set(int(x) for x in np.random.default_rng(seed).choice(available, count, replace=False))
        if set(evaluation) != expected:
            raise ValueError('Stored split differs from requested fraction/seed; regenerate preprocessing')
    return split
