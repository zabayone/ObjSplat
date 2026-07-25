#!/usr/bin/env python3
"""Switch the active ObjSplat day/night PLY without copying scene data."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def switch_mood(scene_root: str | Path, mood: str) -> Path:
    scene_root = Path(scene_root).expanduser().resolve()
    scene_dir = scene_root / "scene"
    manifest_path = scene_dir / "moods.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing mood manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mood_entry = (manifest.get("moods") or {}).get(mood)
    if not mood_entry:
        available = ", ".join(sorted((manifest.get("moods") or {}).keys()))
        raise KeyError(f"Unknown mood '{mood}'. Available: {available}")
    target = scene_root / str(mood_entry["ply_path"])
    if not target.exists():
        raise FileNotFoundError(f"Missing mood PLY: {target}")

    active = scene_dir / "gsplat_scene_active.ply"
    if active.exists() and not active.is_symlink():
        raise RuntimeError(
            f"{active} is a regular file; move it before enabling mood switching"
        )
    relative_target = os.path.relpath(target, active.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=".mood-link-", dir=active.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink()
    try:
        os.symlink(relative_target, tmp_path)
        os.replace(tmp_path, active)
    finally:
        tmp_path.unlink(missing_ok=True)

    manifest["active_mood"] = mood
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    print(f"[mood] active={mood} ply={target}")
    return active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--mood", required=True, choices=["day", "night"])
    args = parser.parse_args()
    switch_mood(args.scene_root, args.mood)


if __name__ == "__main__":
    main()
