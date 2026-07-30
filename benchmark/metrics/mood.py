from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData


def _fields(names, prefixes):
    return [name for prefix in prefixes for name in names if name == prefix or name.startswith(prefix + "_")]


def compare_topology(day_path: str | Path, mood_path: str | Path, tolerance: float = 1e-7) -> dict:
    day = PlyData.read(day_path, mmap="r")["vertex"].data
    mood = PlyData.read(mood_path, mmap="r")["vertex"].data
    result = {
        "day_gaussians": int(len(day)), "mood_gaussians": int(len(mood)),
        "gaussian_count_difference": int(len(mood) - len(day)),
        "correspondence_compatible": len(day) == len(mood),
    }
    if len(day) != len(mood):
        return result
    day_names, mood_names = set(day.dtype.names or ()), set(mood.dtype.names or ())
    if day_names != mood_names:
        result.update(correspondence_compatible=False, reason="PLY property schemas differ")
        return result
    groups = {
        "position": ["x", "y", "z"], "scale": ["scale"], "rotation": ["rot"],
        "opacity": ["opacity"], "sh": ["f_dc", "f_rest"],
    }
    nonappearance_changed = np.zeros(len(day), dtype=bool)
    for group, prefixes in groups.items():
        names = _fields(list(day.dtype.names or ()), prefixes)
        if not names:
            result[f"{group}_mean_abs"] = result[f"{group}_max_abs"] = None
            continue
        total, count, maximum = 0.0, 0, 0.0
        for start in range(0, len(day), 250_000):
            end = min(len(day), start + 250_000)
            delta = np.abs(
                np.column_stack([day[n][start:end] for n in names]).astype(np.float64)
                - np.column_stack([mood[n][start:end] for n in names]).astype(np.float64)
            )
            total += float(delta.sum())
            count += int(delta.size)
            maximum = max(maximum, float(delta.max(initial=0)))
            if group != "sh":
                nonappearance_changed[start:end] |= np.any(delta > tolerance, axis=1)
        result[f"{group}_mean_abs"] = total / max(1, count)
        result[f"{group}_max_abs"] = maximum
    if "label" in day_names:
        result["label_difference_count"] = int(np.count_nonzero(day["label"] != mood["label"]))
        nonappearance_changed |= day["label"] != mood["label"]
    else:
        result["label_difference_count"] = None
    result["nonappearance_changed_percent"] = float(nonappearance_changed.mean() * 100)
    return result
