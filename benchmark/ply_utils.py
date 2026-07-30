from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def inspect_ply(path: str | Path) -> dict:
    path = Path(path)
    from plyfile import PlyData
    ply = PlyData.read(path, mmap="r")
    vertex = ply["vertex"]
    fields = list(vertex.data.dtype.names or ())
    return {
        "path": str(path), "exists": True, "size_bytes": path.stat().st_size,
        "format": "ascii" if ply.text else "binary",
        "byte_order": ply.byte_order, "vertex_count": int(vertex.count),
        "properties": [
            {"name": prop.name, "dtype": str(prop.val_dtype)} for prop in vertex.properties
        ],
        "has_semantic_labels": "label" in fields,
        "sh_degree": infer_sh_degree(fields),
    }


def infer_sh_degree(fields: list[str]) -> int | None:
    rest = len([name for name in fields if name.startswith("f_rest_")])
    if rest == 0:
        return 0 if all(f"f_dc_{i}" in fields for i in range(3)) else None
    coefficients = rest // 3 + 1
    degree = int(round(np.sqrt(coefficients) - 1))
    return degree if (degree + 1) ** 2 == coefficients else None


def read_vertex(path: str | Path, fields: list[str] | None = None):
    from plyfile import PlyData
    data = PlyData.read(path, mmap="r")["vertex"].data
    if fields is None:
        return data
    missing = [field for field in fields if field not in (data.dtype.names or ())]
    if missing:
        raise ValueError(f"{path} missing PLY properties: {missing}")
    return np.column_stack([np.asarray(data[field]) for field in fields])


def filter_ply_by_label(source: str | Path, target: str | Path, labels_to_remove: set[int]) -> dict:
    from plyfile import PlyData, PlyElement
    source, target = Path(source), Path(target)
    ply = PlyData.read(source)
    vertex = ply["vertex"].data
    if "label" not in (vertex.dtype.names or ()):
        raise ValueError("Instance filtering unavailable: PLY has no label property")
    keep = ~np.isin(np.asarray(vertex["label"], dtype=np.int64), list(labels_to_remove))
    target.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex[keep], "vertex")], text=ply.text, byte_order=ply.byte_order).write(target)
    return {
        "source_gaussians": int(len(vertex)), "edited_gaussians": int(keep.sum()),
        "removed_gaussians": int((~keep).sum()), "target": str(target),
    }
