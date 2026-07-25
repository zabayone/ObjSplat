#!/usr/bin/env python3
"""Generate a masked night-sky ERP for an ObjSplat scene."""

from __future__ import annotations

import argparse

from utils.sky_retexture import DEFAULT_NIGHT_PROMPT, SkyRetextureConfig, retexture_sky


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_root", required=True, help="Scene root containing rgb.png and traindata/")
    parser.add_argument(
        "--model_path",
        default="checkpoints/FLUX.1-Fill-dev",
        help="Local FLUX.1 Fill Diffusers checkpoint",
    )
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--prompt", default=DEFAULT_NIGHT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--mask_dilate_px", type=int, default=5)
    parser.add_argument("--mask_feather_px", type=int, default=9)
    parser.add_argument("--circular_padding_ratio", type=float, default=0.0625)
    parser.add_argument("--min_sky_coverage", type=float, default=0.005)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without loading FLUX")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SkyRetextureConfig(
        model_path=args.model_path,
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_pixels=args.max_pixels,
        mask_dilate_px=args.mask_dilate_px,
        mask_feather_px=args.mask_feather_px,
        circular_padding_ratio=args.circular_padding_ratio,
        min_sky_coverage=args.min_sky_coverage,
        device=args.device,
        cpu_offload=not args.no_cpu_offload,
    )
    retexture_sky(
        scene_root=args.scene_root,
        config=config,
        metadata_path=args.metadata_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
