from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from benchmark.config import load_config, validate_config
from benchmark.editing_evaluation import resolve_layer_selectors
from benchmark.instrumentation import BenchmarkRecorder
from benchmark.input_preparation import prepare_panorama
from benchmark.instrumentation.resources import ResourceSampler
from benchmark.io_utils import read_csv
from benchmark.metrics.editing import edit_locality_metrics
from benchmark.metrics.image_metrics import mae, psnr
from benchmark.metrics.mood import compare_topology
from benchmark.metrics.segmentation import mask_metrics
from benchmark.ply_utils import inspect_ply
from benchmark.reporting import aggregate_results, generate_report
from benchmark.system_info import collect_system_info
from benchmark.run_benchmark import (
    _find_reusable_run,
    _link_shared_preprocessing,
    _run_fingerprint,
    expand_scenes,
)
from mps_splat_backend import _balanced_camera_indices
from LayerPano import adaptive_iteration_count, allocate_gaussian_budgets


def tiny_vertex(count=2):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("label", "i4"),
    ]
    value = np.zeros(count, dtype=dtype)
    value["x"] = np.arange(count)
    value["rot_0"] = 1
    value["label"] = np.arange(count)
    return value


class RecorderTests(unittest.TestCase):
    def test_stage_timing_and_incremental_write(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BenchmarkRecorder(directory, "exp", "scene", sample_interval=0.05)
            recorder.start()
            with recorder.stage("work", iterations=3):
                time.sleep(0.02)
            rows = read_csv(Path(directory) / "stage_timings.csv")
            self.assertEqual(rows[0]["status"], "success")
            self.assertGreater(float(rows[0]["wall_seconds"]), 0)
            recorder.finalize()

    def test_failed_stage_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BenchmarkRecorder(directory, "exp", "scene")
            recorder.start()
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with recorder.stage("broken"):
                    raise RuntimeError("simulated")
            recorder.finalize("failed")
            rows = read_csv(Path(directory) / "stage_timings.csv")
            self.assertEqual(rows[0]["status"], "failed")
            summary = json.loads((Path(directory) / "run_summary.json").read_text())
            self.assertEqual(summary["failed_stage"], "broken")

    def test_resource_sampler_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resources.csv"
            sampler = ResourceSampler(path, {"experiment": "e", "scene": "s", "run_id": "r"}, 0.05)
            sampler.start()
            time.sleep(0.12)
            sampler.stop()
            self.assertGreaterEqual(len(read_csv(path)), 1)


class MetricTests(unittest.TestCase):
    def test_image_metrics(self):
        a = np.zeros((8, 8, 3), np.uint8)
        b = a.copy()
        self.assertTrue(np.isinf(psnr(a, b)))
        b[:] = 255
        self.assertAlmostEqual(psnr(a, b), 0.0, places=5)
        self.assertAlmostEqual(mae(a, b), 1.0, places=5)

    def test_segmentation_iou_and_dice(self):
        gt = np.array([[1, 1], [0, 0]], bool)
        pred = np.array([[1, 0], [1, 0]], bool)
        result = mask_metrics(pred, gt)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertAlmostEqual(result["dice"], 0.5)

    def test_edit_locality_definitions(self):
        before = np.zeros((2, 2, 3), np.uint8)
        after = before.copy()
        after[0, 0] = 255
        mask = np.array([[1, 0], [0, 0]], bool)
        result = edit_locality_metrics(before, after, mask)
        self.assertEqual(result["edit_leakage_ratio"], 0)
        self.assertEqual(result["edit_locality_score"], 1)


class ArtifactTests(unittest.TestCase):
    def test_ply_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ply"
            PlyData([PlyElement.describe(tiny_vertex(), "vertex")], text=False).write(path)
            info = inspect_ply(path)
            self.assertEqual(info["vertex_count"], 2)
            self.assertTrue(info["has_semantic_labels"])
            self.assertIn("x", [item["name"] for item in info["properties"]])

    def test_day_night_property_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            day, mood = Path(directory) / "day.ply", Path(directory) / "mood.ply"
            base = tiny_vertex()
            changed = base.copy()
            changed["f_dc_0"] += 0.5
            PlyData([PlyElement.describe(base, "vertex")]).write(day)
            PlyData([PlyElement.describe(changed, "vertex")]).write(mood)
            result = compare_topology(day, mood)
            self.assertTrue(result["correspondence_compatible"])
            self.assertEqual(result["position_max_abs"], 0)
            self.assertGreater(result["sh_max_abs"], 0)
            self.assertEqual(result["nonappearance_changed_percent"], 0)

    def test_system_metadata_collection(self):
        info = collect_system_info(Path(__file__).resolve().parents[2], argv=["test"])
        self.assertIn("git", info)
        self.assertIn("hardware", info)
        self.assertIn("python", info)


class ConfigAndReportTests(unittest.TestCase):
    def test_configuration_validation(self):
        with self.assertRaises(ValueError):
            validate_config({"experiment_name": "", "scenes": []})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "experiment_name": "test", "scenes": [{"scene_root": "scene"}]
            }))
            config = load_config(path)
            self.assertEqual(config["random_seed"], 42)
        with self.assertRaisesRegex(ValueError, "selected_layers"):
            validate_config({
                "experiment_name": "test",
                "scenes": [{"scene_root": "scene"}],
                "selected_layers": ["unknown"],
            })

    def test_global_budget_and_adaptive_iterations(self):
        budgets = allocate_gaussian_budgets(
            {0: 4_000_000, 1: 1_000_000, 2: 40_000},
            total_budget=2_000_000,
            minimum_per_layer=20_000,
        )
        self.assertEqual(sum(budgets.values()), 2_000_000)
        self.assertGreater(budgets[0], budgets[1])
        self.assertGreater(budgets[2] / 40_000, budgets[0] / 4_000_000)
        self.assertLess(
            adaptive_iteration_count(800, 40_000),
            adaptive_iteration_count(800, 1_000_000),
        )

    def test_semantic_layer_size_selectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "traindata").mkdir()
            (root / "scene").mkdir()
            (root / "traindata" / "layer_instances.json").write_text(json.dumps({
                "background_layer_idx": 3,
                "layer_groups": [
                    {"layer_idx": 0}, {"layer_idx": 1},
                    {"layer_idx": 2}, {"layer_idx": 3},
                ],
            }))
            for index, count in enumerate((2, 7, 4, 10)):
                PlyData([
                    PlyElement.describe(tiny_vertex(count), "vertex")
                ]).write(root / "scene" / f"gsplat_layer{index}.ply")
            self.assertEqual(
                resolve_layer_selectors(
                    root, ["smallest", "median", "largest"]
                ),
                [0, 2, 1],
            )

    def test_input_preparation_converts_and_records_source(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpeg"
            Image.new("RGB", (20, 10), (10, 20, 30)).save(source, "JPEG")
            manifest = prepare_panorama(source, root / "work", target_width=512)
            target = root / "work" / "rgb.png"
            with Image.open(target) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (512, 256))
            self.assertEqual(manifest["source_format"], "JPEG")
            self.assertEqual(
                prepare_panorama(source, root / "work", target_width=512)["source_sha256"],
                manifest["source_sha256"],
            )

    def test_ablation_expansion_uses_unique_roots(self):
        config = {
            "scenes": [
                {"name": "indoor", "scene_root": "unused", "pipeline_args": []},
                {"name": "outdoor", "scene_root": "unused2", "pipeline_args": []},
            ],
            "ablations": [{
                "name": "half",
                "scene_root_template": "work/{scene}/half",
                "pipeline_args": ["--downsample_ratio", "0.5"],
            }],
            "shared_preprocessing_root_template": "work/main/{scene}",
        }
        expanded = expand_scenes(config)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(len({item["scene_root"] for item in expanded}), 2)
        self.assertEqual(
            expanded[0]["shared_preprocessing_root"], "work/main/indoor"
        )

    def test_balanced_camera_schedule(self):
        import random

        schedule = _balanced_camera_indices(4, 10, rng=random.Random(7))
        self.assertEqual(len(schedule), 10)
        self.assertEqual(sorted(schedule[:4]), [0, 1, 2, 3])
        self.assertEqual(sorted(schedule[4:8]), [0, 1, 2, 3])
        counts = np.bincount(schedule, minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_successful_run_reuse_and_shared_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "main"
            (shared / "traindata").mkdir(parents=True)
            (shared / "rgb.png").write_bytes(b"rgb")
            (shared / "input_preparation.json").write_text("{}")
            (shared / "traindata" / "layer_instances.json").write_text("{}")
            variant = root / "variant"
            _link_shared_preprocessing(variant, shared)
            self.assertEqual(
                (variant / "traindata").resolve(),
                (shared / "traindata").resolve(),
            )

            config = {"experiment_name": "exp", "scenes": [], "random_seed": 42}
            source = root / "source.jpg"
            source.write_bytes(b"first")
            scene = {
                "name": "scene",
                "scene_root": "work",
                "input_panorama": str(source),
            }
            fingerprint = _run_fingerprint(config, scene)
            source.write_bytes(b"second")
            self.assertNotEqual(fingerprint, _run_fingerprint(config, scene))
            run = root / "results" / "scene" / "run-1"
            run.mkdir(parents=True)
            (run / "run_summary.json").write_text(
                json.dumps({"status": "success"})
            )
            (run / "experiment_config.json").write_text(
                json.dumps({"run_fingerprint": fingerprint})
            )
            self.assertEqual(
                _find_reusable_run(root / "results", "scene", fingerprint),
                run,
            )

    def test_aggregation_with_missing_optional_files_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "experiment" / "scene"
            run.mkdir(parents=True)
            (run / "run_summary.json").write_text(json.dumps({"status": "success"}))
            output = root / "report"
            result = aggregate_results(root / "experiment", output)
            self.assertEqual(result["robustness"]["successful"], 1)
            self.assertTrue((output / "report.md").exists())
            self.assertTrue((output / "plots" / "total_runtime_by_scene.png").exists())
            self.assertEqual(generate_report(output).name, "report.md")
            report = (output / "report.md").read_text()
            self.assertIn("Reconstruction fidelity", report)
            self.assertNotIn("All diagnostic summaries", report)
            stale_report = root / "experiment" / "report"
            stale_report.mkdir()
            (stale_report / "run_summary.json").write_text(
                json.dumps({"status": "success"})
            )
            second_output = root / "second_report"
            repeated = aggregate_results(root / "experiment", second_output)
            self.assertEqual(repeated["robustness"]["scene_runs"], 1)


if __name__ == "__main__":
    unittest.main()
