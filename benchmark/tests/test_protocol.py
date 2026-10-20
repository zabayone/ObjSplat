from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.comparison import write_comparison_metrics
from benchmark.evaluation import evaluate_reconstruction
from benchmark.io_utils import read_csv, write_csv
from benchmark.metrics.image_metrics import mae, psnr
from benchmark.reporting import aggregate_results, paired_variant_summary, scene_mean_rows, statistical_summary
from benchmark.rendering.mlx_renderer import MLXSceneRenderer
from benchmark.schemas import STAGE_COLUMNS, RESOURCE_COLUMNS
from benchmark.validation import validate_view_split


class ProtocolTests(unittest.TestCase):
    def test_dark_uint8_is_scaled_and_float_input_not_mutated(self):
        a = np.zeros((8, 8, 3), np.uint8)
        b = np.ones_like(a)
        self.assertAlmostEqual(mae(a, b), 1 / 255, places=7)
        self.assertAlmostEqual(psnr(a, b), 20 * np.log10(255), places=5)
        values = np.full((8, 8, 3), 128.0, np.float32)
        mae(values, a)
        np.testing.assert_array_equal(values, 128)

    def test_pairing_does_not_mix_runs_or_views(self):
        rows = [
            {'scene': 's', 'run_id': 'a', 'variant': 'layered', 'view_id': 1, 'psnr_db': 10},
            {'scene': 's', 'run_id': 'b', 'variant': 'monolithic', 'view_id': 1, 'psnr_db': 30},
        ]
        self.assertEqual(paired_variant_summary(rows, 'psnr_db')['pair_count'], 0)
        rows[1]['run_id'] = 'a'
        rows[1]['view_id'] = 2
        self.assertEqual(paired_variant_summary(rows, 'psnr_db')['pair_count'], 0)
        rows[1]['view_id'] = 1
        self.assertEqual(paired_variant_summary(rows, 'psnr_db')['difference_summary']['mean'], 20)

    def test_scene_is_the_reconstruction_unit(self):
        rows = [{'scene': 'a', 'variant': 'layered', 'psnr_db': x} for x in (10, 20, 30)]
        rows += [{'scene': 'b', 'variant': 'layered', 'psnr_db': 40}]
        means = scene_mean_rows(rows)
        self.assertEqual(len(means), 2)
        self.assertEqual(np.mean([x['psnr_db'] for x in means]), 30)
        self.assertEqual(statistical_summary([])['count'], 0)
        self.assertIsNone(statistical_summary([1])['std'])

    def split_fixture(self, root):
        frames = root / 'traindata/perspective_frames/frames'
        frames.mkdir(parents=True)
        for index in range(4):
            Image.new('RGB', (8, 8)).save(frames / f'rgb_{index}.png')
            np.save(frames / f'transform_matrix_{index}.npy', np.eye(4))
        metadata = {'benchmark_view_split': {'training_indices': [0, 1, 2], 'evaluation_indices': [3]}}
        (root / 'traindata/layer_instances.json').write_text(json.dumps(metadata))

    def test_split_rejects_training_leakage_and_stale_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.split_fixture(root)
            validate_view_split(root)
            with self.assertRaisesRegex(ValueError, 'fraction/seed'):
                validate_view_split(root, evaluation_fraction=0.25, seed=42)
            layer = root / 'traindata/layer0/frames'
            layer.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(layer / 'rgb_3.png')
            with self.assertRaisesRegex(ValueError, 'leaked'):
                validate_view_split(root)

    def test_no_held_out_result_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'traindata').mkdir()
            (root / 'traindata/layer_instances.json').write_text('{}')
            rows = evaluate_reconstruction(root, 'unused.ply', root / 'out', {}, 'layered')
            self.assertEqual(rows[0]['status'], 'unavailable')
            self.assertEqual(read_csv(root / 'out/reconstruction_metrics.csv')[0]['status'], 'unavailable')

    def test_disabled_metrics_do_not_fail_cost_only_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'stage_timings.csv', [{'stage': 'all_layer_training', 'status': 'success', 'wall_seconds': 3}], STAGE_COLUMNS)
            write_csv(root / 'resource_samples.csv', [{'stage': 'all_layer_training', 'process_rss_bytes': 100}], RESOURCE_COLUMNS)
            rows = write_comparison_metrics(root, root, {}, {'retrain': True}, {'name': 's'})
            self.assertEqual(rows[0]['status'], 'success')
            self.assertIsNone(rows[0]['psnr_db'])

    def test_warmup_frames_are_additional_to_cold_start(self):
        renderer = MLXSceneRenderer.__new__(MLXSceneRenderer)
        renderer.gaussian_count = 1
        from unittest.mock import Mock
        renderer.mx = Mock()
        renderer._render_mlx = Mock(return_value=0)
        renderer.benchmark([np.eye(4)], 8, 8, warmup=3, measured=2)
        self.assertEqual(renderer._render_mlx.call_count, 6)
        with self.assertRaises(ValueError):
            renderer.benchmark([np.eye(4)], 8, 8, 1, 0)

    def test_aggregation_uses_explicit_run_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run in ('old', 'new'):
                path = root / 's' / run
                path.mkdir(parents=True)
                (path / 'run_summary.json').write_text(json.dumps({'status': 'success', 'run_id': run}))
            (root / 'selected_runs.json').write_text(json.dumps([{'output': str(root / 's/new')}]))
            with patch('benchmark.reporting.generate_plots'), patch('benchmark.reporting.generate_report'):
                result = aggregate_results(root, root / 'report')
            self.assertEqual(result['robustness']['scene_runs'], 1)
            self.assertEqual(result['runs'][0]['run_id'], 'new')


if __name__ == '__main__':
    unittest.main()
