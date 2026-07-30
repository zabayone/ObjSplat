from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from benchmark.io_utils import append_csv
from benchmark.schemas import RESOURCE_COLUMNS

try:
    import psutil
except ImportError:  # optional
    psutil = None


def memory_snapshot() -> dict:
    if psutil is None:
        return {
            "process_rss_bytes": None, "process_vms_bytes": None,
            "system_total_bytes": None, "system_available_bytes": None,
            "system_used_bytes": None, "system_used_percent": None,
            "swap_total_bytes": None, "swap_used_bytes": None,
            "process_cpu_percent": None,
        }
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    child_mem = []
    try:
        child_mem = [child.memory_info() for child in process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "process_rss_bytes": int(mem.rss + sum(item.rss for item in child_mem)),
        "process_vms_bytes": int(mem.vms + sum(item.vms for item in child_mem)),
        "system_total_bytes": int(vm.total),
        "system_available_bytes": int(vm.available),
        "system_used_bytes": int(vm.used),
        "system_used_percent": float(vm.percent),
        "swap_total_bytes": int(swap.total),
        "swap_used_bytes": int(swap.used),
        "process_cpu_percent": float(process.cpu_percent(interval=None)),
    }


class ResourceSampler:
    def __init__(
        self, output: str | Path, metadata: dict, interval: float = 1.0,
        stage_getter: Callable[[], str | None] | None = None,
    ):
        self.output = Path(output)
        self.metadata = metadata
        self.interval = max(0.1, float(interval))
        self.stage_getter = stage_getter or (lambda: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0
        self.last_sample: dict | None = None
        self.peak_rss = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="benchmark-resource-sampler", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        row = {
            **self.metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self._start,
            "stage": self.stage_getter(),
            **memory_snapshot(),
        }
        rss = row.get("process_rss_bytes")
        if rss is not None:
            self.peak_rss = max(self.peak_rss, int(rss))
        self.last_sample = row
        append_csv(self.output, row, RESOURCE_COLUMNS)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 2))
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.stop()
