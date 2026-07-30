from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def append_csv(path: str | Path, row: dict[str, Any], columns: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: json_safe(row.get(key)) for key in columns})
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], columns: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json_safe(row.get(key)) for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
