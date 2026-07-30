#!/usr/bin/env python3
"""Launch the SuperSplat viewer for a local .ply file.

The script starts a small HTTP server for the file's directory, launches the
GUI dev server if it is not already running, and opens the browser with the
correct `load` and `filename` query parameters.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _resolve_gui_dir() -> Path:
    candidates = [
        REPO_ROOT / "GUI",
        SCRIPT_DIR / "GUI",
    ]
    for candidate in candidates:
        if (candidate / "package.json").exists():
            return candidate
    # Keep the canonical path in the error message for easier debugging.
    return REPO_ROOT / "GUI"


GUI_DIR = _resolve_gui_dir()
VIEWER_URL = "http://127.0.0.1:3000/"


def is_http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except Exception:
        return False


def wait_for_http(url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_http_ready(url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class QuietFileHandler(SimpleHTTPRequestHandler):
    _byte_range: tuple[int, int] | None = None

    def log_message(self, format: str, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header(
            "Access-Control-Expose-Headers",
            "Content-Range, Content-Length, Accept-Ranges",
        )
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def send_head(self):
        """Serve a single byte range without buffering the complete PLY."""
        self._byte_range = None
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        try:
            stat = os.fstat(source.fileno())
            size = int(stat.st_size)
            content_type = self.guess_type(path)
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if match is None:
                    self.send_error(416, "Only one byte range is supported")
                    source.close()
                    return None
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                elif last:
                    suffix_length = int(last)
                    start = max(0, size - suffix_length)
                    end = size - 1
                else:
                    self.send_error(416, "Invalid byte range")
                    source.close()
                    return None
                if start >= size or start < 0 or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    source.close()
                    return None
                end = min(end, size - 1)
                self._byte_range = (start, end)
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
                self.end_headers()
                source.seek(start)
                return source

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return source
        except Exception:
            source.close()
            raise

    def copyfile(self, source, outputfile):
        try:
            if self._byte_range is None:
                super().copyfile(source, outputfile)
                return
            start, end = self._byte_range
            remaining = end - start + 1
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_viewer_if_needed() -> subprocess.Popen[str] | None:
    if not (GUI_DIR / "package.json").exists():
        raise FileNotFoundError(f"Cannot find SuperSplat GUI project at {GUI_DIR}")

    if is_http_ready(VIEWER_URL):
        return None

    subprocess.run(
        ["npm", "run", "build"],
        cwd=str(GUI_DIR),
        check=True,
    )

    return subprocess.Popen(
        ["npm", "run", "serve"],
        cwd=str(GUI_DIR),
    )


def start_file_server(directory: Path) -> tuple[ThreadingHTTPServer, threading.Thread, int]:
    port = find_free_port()
    handler = functools.partial(QuietFileHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, port


def build_viewer_url(
    file_url: str,
    filename: str,
    cache_bust: str,
    mood_manifest_url: str | None = None,
    mood_root_url: str | None = None,
) -> str:
    params = {
        "load": file_url,
        "filename": filename,
        "localViewer": "1",
        "cacheBust": cache_bust,
    }
    if mood_manifest_url:
        params["moodManifest"] = mood_manifest_url
    if mood_root_url:
        params["moodRoot"] = mood_root_url
    query = urllib.parse.urlencode(params)
    return f"{VIEWER_URL}?{query}"


def resolve_scene_context(input_path: Path) -> tuple[Path, Path, Path | None]:
    """Resolve the selected PLY, HTTP root and optional mood manifest."""
    input_path = input_path.expanduser().resolve()
    manifest_path: Path | None = None
    scene_root: Path | None = None

    if input_path.is_dir():
        if (input_path / "scene" / "moods.json").exists():
            scene_root = input_path
            manifest_path = input_path / "scene" / "moods.json"
        elif input_path.name == "scene" and (input_path / "moods.json").exists():
            scene_root = input_path.parent
            manifest_path = input_path / "moods.json"
        else:
            raise FileNotFoundError(
                f"Directory does not contain scene/moods.json: {input_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        moods = manifest.get("moods") or {}
        active_name = str(manifest.get("active_mood") or "day")
        active_entry = moods.get(active_name) or moods.get("day")
        if not active_entry or not active_entry.get("ply_path"):
            raise ValueError(f"Mood manifest has no loadable active mood: {manifest_path}")
        ply_path = (scene_root / str(active_entry["ply_path"])).resolve()
    else:
        ply_path = input_path
        if ply_path.parent.name == "scene":
            candidate = ply_path.parent / "moods.json"
            if candidate.exists():
                scene_root = ply_path.parent.parent
                manifest_path = candidate

    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if ply_path.suffix.lower() != ".ply":
        raise ValueError("The input file must have a .ply extension")

    server_root = scene_root if scene_root is not None else ply_path.parent
    try:
        ply_path.relative_to(server_root)
    except ValueError as exc:
        raise ValueError("PLY must be inside the local scene root") from exc
    return ply_path, server_root, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open a PLY or mood-enabled ObjSplat scene in SuperSplat."
    )
    parser.add_argument(
        "scene_path",
        type=Path,
        help="Path to a .ply file, ObjSplat scene root, or scene/ directory",
    )
    args = parser.parse_args()

    ply_path, server_root, manifest_path = resolve_scene_context(args.scene_path)

    file_server, file_server_thread, file_port = start_file_server(server_root)
    viewer_process = start_viewer_if_needed()

    try:
        wait_for_http(VIEWER_URL, timeout_seconds=180)
        cache_bust = str(int(time.time() * 1000))
        relative_ply = ply_path.relative_to(server_root).as_posix()
        file_url = (
            f"http://127.0.0.1:{file_port}/"
            f"{urllib.parse.quote(relative_ply, safe='/')}?v={cache_bust}"
        )
        mood_manifest_url = None
        mood_root_url = None
        if manifest_path is not None:
            relative_manifest = manifest_path.relative_to(server_root).as_posix()
            mood_manifest_url = (
                f"http://127.0.0.1:{file_port}/"
                f"{urllib.parse.quote(relative_manifest, safe='/')}?v={cache_bust}"
            )
            mood_root_url = f"http://127.0.0.1:{file_port}/"
        viewer_url = build_viewer_url(
            file_url,
            ply_path.name,
            cache_bust,
            mood_manifest_url=mood_manifest_url,
            mood_root_url=mood_root_url,
        )
        print(f"Opening SuperSplat: {viewer_url}")
        webbrowser.open(viewer_url, new=1)
        print("Keep this terminal open until the file finishes loading. Press Ctrl+C to stop the local file server.")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        file_server.shutdown()
        file_server.server_close()
        file_server_thread.join(timeout=2)
        if viewer_process is not None:
            viewer_process.terminate()
            try:
                viewer_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                viewer_process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
