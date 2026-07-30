from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np


class MLXSceneRenderer:
    """Thin reusable adapter around ObjSplat's MLX renderer."""

    def __init__(self, ply_path: str | Path, rasterizer: str = "cpp"):
        repo = Path(__file__).resolve().parents[2]
        splat_apple = repo / "submodules" / "splat-apple"
        if str(splat_apple) not in sys.path:
            sys.path.insert(0, str(splat_apple))
        import mlx.core as mx
        from mlx_gs.renderer.renderer import render
        from mlx_gs.training.trainer import Camera
        from mps_splat_backend import extract_gaussian_params_from_ply
        self.mx = mx
        self._render = render
        self._camera_type = Camera
        params, labels = extract_gaussian_params_from_ply(str(ply_path))
        self.params = {key: mx.array(value, dtype=mx.float32) for key, value in params.items()}
        self.labels = labels
        self.rasterizer = rasterizer
        self.gaussian_count = int(len(params["means"]))

    def _render_mlx(
        self, transform_matrix, width: int, height: int, fov_degrees: float = 90.0
    ):
        from mps_splat_backend import _pose_to_w2c
        fovx = math.radians(float(fov_degrees))
        fovy = 2.0 * math.atan((height / width) * math.tan(fovx / 2.0))
        camera = self._camera_type(
            W=int(width), H=int(height),
            fx=width / (2 * math.tan(fovx / 2)),
            fy=height / (2 * math.tan(fovy / 2)),
            cx=width / 2, cy=height / 2,
            W2C=self.mx.array(_pose_to_w2c(np.asarray(transform_matrix, dtype=np.float32))),
        )
        return self._render(
            self.params,
            camera,
            rasterizer_type=self.rasterizer,
            active_sh_degree=3,
        )

    def render(self, transform_matrix, width: int, height: int, fov_degrees: float = 90.0) -> np.ndarray:
        image = self._render_mlx(transform_matrix, width, height, fov_degrees)
        self.mx.eval(image)
        return np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)

    def benchmark(self, poses, width: int, height: int, warmup: int, measured: int) -> dict:
        poses = list(poses)
        if not poses:
            raise ValueError("At least one camera pose is required")
        cold_start = time.perf_counter()
        image = self._render_mlx(poses[0], width, height)
        self.mx.eval(image)
        cold_seconds = time.perf_counter() - cold_start
        for index in range(max(0, warmup - 1)):
            image = self._render_mlx(poses[index % len(poses)], width, height)
            self.mx.eval(image)
        timings = []
        for index in range(measured):
            started = time.perf_counter()
            image = self._render_mlx(poses[index % len(poses)], width, height)
            self.mx.eval(image)
            timings.append((time.perf_counter() - started) * 1000)
        values = np.asarray(timings, dtype=np.float64)
        return {
            "cold_start_seconds": cold_seconds, "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90)),
            "p95_ms": float(np.percentile(values, 95)),
            "average_fps": float(1000 / values.mean()),
            "minimum_fps": float(1000 / values.max()),
            "megapixels_per_second": float(width * height / 1e6 * 1000 / values.mean()),
            "gaussian_count": self.gaussian_count,
        }
