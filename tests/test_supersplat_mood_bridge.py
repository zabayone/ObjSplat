from __future__ import annotations

import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from utils.open_ply_in_supersplat import build_viewer_url, resolve_scene_context


class SuperSplatMoodBridgeTests(unittest.TestCase):
    def test_scene_directory_resolves_active_mood(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scene = root / "scene"
            scene.mkdir()
            day = scene / "day.ply"
            serene = scene / "serene.ply"
            day.touch()
            serene.touch()
            (scene / "moods.json").write_text(
                json.dumps(
                    {
                        "active_mood": "serene",
                        "moods": {
                            "day": {"ply_path": "scene/day.ply"},
                            "serene": {"ply_path": "scene/serene.ply"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            ply, server_root, manifest = resolve_scene_context(root)
            self.assertEqual(ply, serene)
            self.assertEqual(server_root, root)
            self.assertEqual(manifest, scene / "moods.json")

    def test_ply_inside_scene_discovers_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scene = root / "scene"
            scene.mkdir()
            ply = scene / "day.ply"
            ply.touch()
            manifest = scene / "moods.json"
            manifest.write_text('{"moods": {}}', encoding="utf-8")

            resolved, server_root, discovered = resolve_scene_context(ply)
            self.assertEqual(resolved, ply)
            self.assertEqual(server_root, root)
            self.assertEqual(discovered, manifest)

    def test_viewer_url_contains_mood_bridge_parameters(self) -> None:
        url = build_viewer_url(
            "http://127.0.0.1:8123/scene/day.ply",
            "day.ply",
            "123",
            mood_manifest_url="http://127.0.0.1:8123/scene/moods.json",
            mood_root_url="http://127.0.0.1:8123/",
        )
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(query["filename"], ["day.ply"])
        self.assertEqual(
            query["moodManifest"],
            ["http://127.0.0.1:8123/scene/moods.json"],
        )
        self.assertEqual(query["moodRoot"], ["http://127.0.0.1:8123/"])


if __name__ == "__main__":
    unittest.main()
