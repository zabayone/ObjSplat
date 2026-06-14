#!/usr/bin/env python3
"""Launch the SuperSplat viewer for a local .ply file.

The script starts a small HTTP server for the file's directory, launches the
GUI dev server if it is not already running, and opens the browser with the
correct `load` and `filename` query parameters.
"""

from __future__ import annotations

import argparse
import functools
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
    def log_message(self, format: str, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except BrokenPipeError:
            pass


def start_viewer_if_needed() -> subprocess.Popen[str] | None:
    if not (GUI_DIR / "package.json").exists():
        raise FileNotFoundError(f"Cannot find SuperSplat GUI project at {GUI_DIR}")

    subprocess.run(
        ["npm", "run", "build"],
        cwd=str(GUI_DIR),
        check=True,
    )

    if is_http_ready(VIEWER_URL):
        return None

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


def build_viewer_url(file_url: str, filename: str, cache_bust: str) -> str:
    query = urllib.parse.urlencode({
        "load": file_url,
        "filename": filename,
        "localViewer": "1",
        "cacheBust": cache_bust,
    })
    return f"{VIEWER_URL}?{query}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a .ply file in SuperSplat.")
    parser.add_argument("ply_path", type=Path, help="Path to the .ply file")
    args = parser.parse_args()

    ply_path = args.ply_path.expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if ply_path.suffix.lower() != ".ply":
        raise ValueError("The input file must have a .ply extension")

    file_server, file_server_thread, file_port = start_file_server(ply_path.parent)
    viewer_process = start_viewer_if_needed()

    try:
        wait_for_http(VIEWER_URL, timeout_seconds=180)
        cache_bust = str(int(time.time() * 1000))
        file_url = f"http://127.0.0.1:{file_port}/{urllib.parse.quote(ply_path.name)}?v={cache_bust}"
        viewer_url = build_viewer_url(file_url, ply_path.name, cache_bust)
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
