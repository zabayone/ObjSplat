"""
Thin Python wrapper around the torch_gs C++ rasterizer extension.

Fixes vs original:
  - render_tiles_cpp: missing closing ) on cpp_gs.render_tiles(...) call
  - get_tile_interactions_cpp: was swallowed as an argument (not defined as a function)
    due to the unclosed parenthesis above; now correctly defined + closed
"""

import torch
import torch_gs._C as cpp_gs


def render_tiles_cpp(
    means2D,
    cov2D,
    opacities,
    colors,
    sorted_tile_ids,
    sorted_gaussian_ids,
    H: int,
    W: int,
    tile_size: int,
    background=None,
    device: str = "mps",
):
    if background is None:
        background = torch.zeros(3, device=device)

    return cpp_gs.render_tiles(
        means2D,
        cov2D,
        opacities,
        colors,
        sorted_tile_ids,
        sorted_gaussian_ids,
        H,
        W,
        tile_size,
        background,
    )


def get_tile_interactions_cpp(
    means2D,
    radii,
    valid_mask,
    depths,
    H: int,
    W: int,
    tile_size: int,
    device: str = "mps",
):
    return cpp_gs.get_tile_interactions(
        means2D,
        radii,
        valid_mask,
        depths,
        H,
        W,
        tile_size,
    )
