"""Regression tests for frozen training and Gaussian serialization."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import mps_splat_backend as backend


class GaussianSerializationTests(unittest.TestCase):
    def test_sh_roundtrip_and_frozen_export(self):
        for degree in (0, 1, 3):
            n = 2
            params = {
                'means': np.ones((n, 3), np.float32),
                'scales': np.full((n, 3), -12, np.float32),
                'quaternions': np.tile(np.array([2, 0, 0, 0], np.float32), (n, 1)),
                'opacities': np.full((n, 1), -8, np.float32),
                'sh_coeffs': np.arange(n * (degree + 1)**2 * 3, dtype=np.float32).reshape(n, -1, 3),
            }
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / 'scene.ply')
                backend._save_layerpano_compatible_ply_mlx(
                    path, params, np.array([1, 2]), sh_degree=degree,
                    freeze_geometry=True, freeze_opacity=True,
                )
                restored, labels = backend.extract_gaussian_params_from_ply(path)
            for key in params:
                np.testing.assert_array_equal(restored[key], params[key])
            np.testing.assert_array_equal(labels, [1, 2])

    def test_empty_downsample_returns_indices(self):
        result = backend._voxel_downsample(np.empty((0, 3)), np.empty((0, 3)), None, return_indices=True)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[3].size, 0)


class FrozenTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            backend._ensure_mlx_gs_importable()
            import mlx.core as mx
            import mlx.optimizers as optim
            from mlx_gs.training import trainer
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.mx, cls.optim, cls.trainer = mx, optim, trainer

    def parameters(self):
        mx = self.mx
        return {
            'means': mx.ones((2, 3)), 'scales': mx.full((2, 3), -12.0),
            'quaternions': mx.full((2, 4), 2.0),
            'opacities': mx.full((2, 1), -8.0),
            'sh_coeffs': mx.ones((2, 4, 3)),
        }

    def test_only_selected_arrays_are_differentiated_and_changed(self):
        mx = self.mx
        for selected in ('sh_coeffs', 'opacities'):
            params = self.parameters()
            original = {k: np.array(v) for k, v in params.items()}
            optimizers = {selected: self.optim.Adam(learning_rate=0.01)}
            differentiated = []
            real_value_and_grad = mx.value_and_grad

            def capture(fn):
                transformed = real_value_and_grad(fn)
                def call(p):
                    differentiated.append(set(p))
                    return transformed(p)
                return call

            def fake_loss(p, *args, **kwargs):
                value = sum(mx.sum(v * v) for v in p.values())
                return value, mx.ones((4, 4, 3))

            with patch.object(self.trainer, 'loss_fn', fake_loss), patch.object(mx, 'value_and_grad', capture):
                self.trainer.train_step(params, optimizers, mx.zeros((4, 4, 3)), None, opacity_reg_weight=0)
            mx.eval(params)
            self.assertEqual(differentiated, [{selected}])
            self.assertFalse(np.array_equal(np.array(params[selected]), original[selected]))
            for key in params.keys() - {selected}:
                np.testing.assert_array_equal(np.array(params[key]), original[key])

    def test_initial_refinement_preserves_frozen_attributes_and_accepts_default_cap(self):
        params = {k: np.array(v) for k, v in self.parameters().items()}
        data = {
            'W': 4, 'H': 4, 'fov': 60,
            'pcd_points': params['means'], 'pcd_colors': np.ones((2, 3), np.float32),
            'pcd_labels': np.array([1, 2]),
            'frames': [{'image': Image.new('RGB', (4, 4)), 'transform_matrix': np.eye(4)}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'refined.ply')
            backend.train_with_splat_apple(
                data, path, 0, initial_gaussian_params=params,
                initial_gaussian_labels=np.array([1, 2]), downsample_ratio=1,
                freeze_geometry=True, freeze_opacity=True,
            )
            result, _ = backend.extract_gaussian_params_from_ply(path)
        for key in ('means', 'scales', 'quaternions', 'opacities'):
            np.testing.assert_array_equal(result[key], params[key])

    def test_compact_sh_matches_padded_renderer(self):
        from mlx_gs.renderer.renderer import _eval_sh_rgb
        mx = self.mx
        compact = mx.array(np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 100)
        padded = mx.concatenate([compact, mx.zeros((2, 12, 3))], axis=1)
        directions = mx.array([[0.1, 0.2, 0.3], [-0.1, 0.4, 0.2]])
        for degree in (0, 1, 2, 3):
            np.testing.assert_array_equal(
                np.array(_eval_sh_rgb(compact, directions, degree)),
                np.array(_eval_sh_rgb(padded, directions, degree)),
            )


if __name__ == '__main__':
    unittest.main()
