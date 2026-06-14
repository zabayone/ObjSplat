#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    legacy_script = Path(__file__).resolve().parent / "legacy" / "labelgs_gui.py"
    if not legacy_script.exists():
        raise FileNotFoundError(f"Missing legacy GUI entrypoint: {legacy_script}")

    sys.argv[0] = str(legacy_script)
    runpy.run_path(str(legacy_script), run_name="__main__")


if __name__ == "__main__":
    main()