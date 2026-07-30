"""Opt-in structured pipeline timing activated by OBJSPLAT_BENCHMARK_DIR."""
from __future__ import annotations

import contextlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark.instrumentation.resources import memory_snapshot
from benchmark.io_utils import append_csv
from benchmark.schemas import STAGE_COLUMNS

_STACK: list[str] = []


def enabled() -> bool:
    return bool(os.environ.get("OBJSPLAT_BENCHMARK_DIR"))


def _base() -> dict:
    return {
        "experiment": os.environ.get("OBJSPLAT_BENCHMARK_EXPERIMENT"),
        "scene": os.environ.get("OBJSPLAT_BENCHMARK_SCENE"),
        "run_id": os.environ.get("OBJSPLAT_BENCHMARK_RUN_ID"),
    }


def record_stage(
    name: str, wall_seconds: float, *, cpu_seconds: float | None = None,
    status: str = "success", parent_stage: str | None = None, **counters,
) -> None:
    if not enabled():
        return
    now = datetime.now(timezone.utc).isoformat()
    row = {
        **_base(), "stage": name, "parent_stage": parent_stage,
        "started_at": None, "ended_at": now, "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds, "status": status, **counters,
    }
    _add_rates(row)
    append_csv(Path(os.environ["OBJSPLAT_BENCHMARK_DIR"]) / "stage_timings.csv", row, STAGE_COLUMNS)


@contextlib.contextmanager
def pipeline_stage(name: str, **counters):
    if not enabled():
        yield
        return
    parent = _STACK[-1] if _STACK else None
    _STACK.append(name)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    before = memory_snapshot()
    started_at = datetime.now(timezone.utc).isoformat()
    status, exc_type, exc_message = "success", None, None
    try:
        yield
    except BaseException as exc:
        status, exc_type, exc_message = "failed", type(exc).__name__, str(exc)
        raise
    finally:
        after = memory_snapshot()
        row = {
            **_base(), "stage": name, "parent_stage": parent,
            "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start, "status": status,
            "rss_before_bytes": before.get("process_rss_bytes"),
            "rss_after_bytes": after.get("process_rss_bytes"),
            "system_available_before_bytes": before.get("system_available_bytes"),
            "system_available_after_bytes": after.get("system_available_bytes"),
            "exception_type": exc_type, "exception_message": exc_message, **counters,
        }
        _add_rates(row)
        append_csv(Path(os.environ["OBJSPLAT_BENCHMARK_DIR"]) / "stage_timings.csv", row, STAGE_COLUMNS)
        if _STACK and _STACK[-1] == name:
            _STACK.pop()


def _add_rates(row: dict) -> None:
    wall = float(row.get("wall_seconds") or 0)
    for denominator, output, scale in (
        ("iterations", "seconds_per_iteration", 1.0),
        ("frames", "seconds_per_frame", 1.0),
        ("input_points", "seconds_per_million_input_points", 1_000_000.0),
        ("output_gaussians", "seconds_per_million_output_gaussians", 1_000_000.0),
    ):
        value = row.get(denominator)
        row[output] = wall * scale / float(value) if value not in (None, 0, "0") else None
