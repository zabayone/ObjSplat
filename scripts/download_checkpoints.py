#!/usr/bin/env python3
"""Scarica checkpoint nella cartella `checkpoints/`.

Uso:
  python scripts/download_checkpoints.py --url <URL> --out checkpoints/name.pth
  python scripts/download_checkpoints.py --hf_repo <repo_id> --filename <file> --out checkpoints/name.pth

Lo script prova prima a usare `huggingface_hub` se disponibile, altrimenti `requests`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

def download_via_requests(url: str, out: Path):
    try:
        import requests
    except ImportError:
        raise RuntimeError('requests not installed; pip install requests')
    print(f'Downloading {url} → {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out, 'wb') as fh:
        for chunk in r.iter_content(1024*1024):
            if chunk:
                fh.write(chunk)
    print('Done')

def download_from_hf(repo_id: str, filename: str, out: Path):
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        raise RuntimeError('huggingface_hub not installed; pip install huggingface_hub')
    print(f'Downloading {repo_id}/{filename} → {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(out.parent), local_dir=str(out.parent), local_dir_use_symlinks=False)
    print('Done (saved to cache directory)')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', help='Direct URL to download')
    p.add_argument('--hf_repo', help='HuggingFace repo id (e.g. IDEA-Research/GroundingDINO)')
    p.add_argument('--filename', help='Filename in HF repo')
    p.add_argument('--out', required=True, help='Output path under checkpoints/')
    args = p.parse_args()

    out = Path(args.out)
    if args.hf_repo and args.filename:
        download_from_hf(args.hf_repo, args.filename, out)
    elif args.url:
        download_via_requests(args.url, out)
    else:
        print('Pass either --url or --hf_repo + --filename')

if __name__ == '__main__':
    main()
