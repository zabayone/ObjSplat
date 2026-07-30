from __future__ import annotations

import contextlib
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmark.instrumentation.resources import ResourceSampler, memory_snapshot
from benchmark.io_utils import append_csv, atomic_json
from benchmark.schemas import STAGE_COLUMNS


class BenchmarkRecorder:
    """Exception-safe, incremental stage and resource recorder."""

    def __init__(
        self, output_dir: str | Path, experiment: str, scene: str,
        run_id: str | None = None, sample_interval: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.scene = scene
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._stack: list[str] = []
        self._started_perf = time.perf_counter()
        self._stage_peak_start: dict[str, int] = {}
        base = {"experiment": experiment, "scene": scene, "run_id": self.run_id}
        self.sampler = ResourceSampler(
            self.output_dir / "resource_samples.csv", base, sample_interval,
            stage_getter=lambda: self.current_stage,
        )
        self.summary = {
            **base, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "completed_stages": [], "failed_stage": None,
        }
        atomic_json(self.output_dir / "run_summary.json", self.summary)

    @property
    def current_stage(self) -> str | None:
        return self._stack[-1] if self._stack else None

    def start(self) -> None:
        self.sampler.start()

    def stop(self) -> None:
        self.sampler.stop()

    @contextlib.contextmanager
    def stage(self, name: str, **counters):
        parent = self.current_stage
        started_at = datetime.now(timezone.utc).isoformat()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        before = memory_snapshot()
        self._stack.append(name)
        peak_before = self.sampler.peak_rss
        status = "success"
        exc_type = exc_message = None
        try:
            yield
        except BaseException as exc:
            status = "failed"
            exc_type = type(exc).__name__
            exc_message = str(exc)
            self.summary["failed_stage"] = name
            self.summary["status"] = "failed"
            raise
        finally:
            after = memory_snapshot()
            ended_at = datetime.now(timezone.utc).isoformat()
            row = {
                "experiment": self.experiment, "scene": self.scene, "run_id": self.run_id,
                "stage": name, "parent_stage": parent, "started_at": started_at,
                "ended_at": ended_at, "wall_seconds": time.perf_counter() - wall_start,
                "cpu_seconds": time.process_time() - cpu_start, "status": status,
                "rss_before_bytes": before.get("process_rss_bytes"),
                "rss_after_bytes": after.get("process_rss_bytes"),
                "system_available_before_bytes": before.get("system_available_bytes"),
                "system_available_after_bytes": after.get("system_available_bytes"),
                "peak_sampled_rss_bytes": max(0, self.sampler.peak_rss, peak_before),
                "exception_type": exc_type, "exception_message": exc_message,
                **counters,
            }
            wall = float(row["wall_seconds"])
            row["seconds_per_iteration"] = wall / float(row["iterations"]) if row.get("iterations") else None
            row["seconds_per_frame"] = wall / float(row["frames"]) if row.get("frames") else None
            row["seconds_per_million_input_points"] = (
                wall * 1_000_000 / float(row["input_points"]) if row.get("input_points") else None
            )
            row["seconds_per_million_output_gaussians"] = (
                wall * 1_000_000 / float(row["output_gaussians"]) if row.get("output_gaussians") else None
            )
            append_csv(self.output_dir / "stage_timings.csv", row, STAGE_COLUMNS)
            if self._stack and self._stack[-1] == name:
                self._stack.pop()
            self.summary["completed_stages"].append({"stage": name, "status": status})
            atomic_json(self.output_dir / "run_summary.json", self.summary)

    def finalize(self, status: str = "success", **extra) -> None:
        self.stop()
        self.summary.update(extra)
        self.summary["status"] = status
        self.summary["ended_at"] = datetime.now(timezone.utc).isoformat()
        self.summary["elapsed_seconds"] = time.perf_counter() - self._started_perf
        self.summary["peak_process_rss_bytes"] = self.sampler.peak_rss or None
        self.summary["last_resource_sample"] = self.sampler.last_sample
        atomic_json(self.output_dir / "run_summary.json", self.summary)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, _tb):
        self.finalize(
            "failed" if exc else "success",
            exception_type=exc_type.__name__ if exc_type else None,
            exception_message=str(exc) if exc else None,
        )
