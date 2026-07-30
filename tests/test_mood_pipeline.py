from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from utils.mood_adaptation import (
    adapt_gaussian_ply_to_erp,
    build_mood_scene_erp,
    get_mood_preset,
    mood_from_circumplex,
)
from utils.sky_retexture import (
    SkyRetextureConfig,
    _add_procedural_stars,
    _compress_sky_hotspots,
)


class MoodPipelineTests(unittest.TestCase):
    def test_circumplex_anchor_and_continuous_mood(self) -> None:
        joyful = mood_from_circumplex("joyful_probe", 0.82, 0.74)
        preset = get_mood_preset("joyful")
        self.assertAlmostEqual(joyful.exposure_ev, preset.exposure_ev)
        self.assertAlmostEqual(joyful.contrast, preset.contrast)

        focused = mood_from_circumplex("focused", -0.2, 0.65)
        self.assertEqual(focused.name, "focused")
        self.assertEqual(focused.time_of_day, "day")
        self.assertGreater(focused.contrast, 1.0)

    def test_builds_multiple_mood_erps_without_overwriting_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            sky_dir = root / "traindata" / "sky"
            sky_dir.mkdir(parents=True)
            source = np.zeros((32, 64, 3), dtype=np.uint8)
            source[:16] = [95, 165, 225]
            source[16:] = [170, 130, 75]
            mask = np.zeros((32, 64), dtype=np.uint8)
            mask[:16] = 255
            Image.fromarray(source).save(root / "rgb.png")
            Image.fromarray(mask).save(sky_dir / "mask.png")
            metadata_path = root / "traindata" / "layer_instances.json"
            metadata_path.write_text(
                json.dumps({"sky": {"mask_path": "traindata/sky/mask.png"}}),
                encoding="utf-8",
            )

            for config in (
                get_mood_preset("serene"),
                mood_from_circumplex("focused", -0.2, 0.65),
            ):
                output = build_mood_scene_erp(root, config)
                self.assertTrue(output.exists())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(metadata["moods"]), ["focused", "serene"])

    def test_hotspot_compression_and_stars_are_mask_limited(self) -> None:
        image = np.full((96, 192, 3), [24, 34, 56], dtype=np.uint8)
        image[:, 128:176] = [110, 120, 145]
        mask = np.ones((96, 192), dtype=bool)
        mask[:, :12] = False
        config = SkyRetextureConfig(model_path="unused")

        compressed, report = _compress_sky_hotspots(image, mask, config)
        self.assertGreater(report["adjusted_pixels"], 0)
        self.assertLess(
            compressed[:, 150].astype(np.float32).mean(),
            image[:, 150].astype(np.float32).mean(),
        )
        starred, count = _add_procedural_stars(compressed, mask, config)
        self.assertGreater(count, 0)
        np.testing.assert_array_equal(starred[:, :12], image[:, :12])

    def test_gaussian_adaptation_preserves_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_ply = root / "source.ply"
            target_ply = root / "target.ply"
            erp_path = root / "mood.png"
            mask_path = root / "mask.png"
            dtype = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("f_dc_0", "f4"),
                ("f_dc_1", "f4"),
                ("f_dc_2", "f4"),
                ("f_rest_0", "f4"),
                ("opacity", "f4"),
            ]
            vertices = np.zeros(3, dtype=dtype)
            vertices["x"] = [-1.0, 0.0, 1.0]
            vertices["z"] = [1.0, 1.0, 1.0]
            vertices["opacity"] = [0.2, 0.4, 0.6]
            PlyData(
                [PlyElement.describe(vertices, "vertex")],
                text=False,
            ).write(source_ply)
            Image.fromarray(
                np.full((16, 32, 3), [40, 80, 160], dtype=np.uint8)
            ).save(erp_path)
            Image.fromarray(np.full((16, 32), 255, dtype=np.uint8)).save(mask_path)

            adapt_gaussian_ply_to_erp(
                source_ply,
                target_ply,
                erp_path,
                mask_path,
                get_mood_preset("melancholic"),
            )
            source = PlyData.read(source_ply)["vertex"].data
            target = PlyData.read(target_ply)["vertex"].data
            for field in ("x", "y", "z", "opacity"):
                np.testing.assert_array_equal(source[field], target[field])
            self.assertFalse(np.array_equal(source["f_dc_2"], target["f_dc_2"]))


if __name__ == "__main__":
    unittest.main()
