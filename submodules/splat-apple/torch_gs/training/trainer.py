"""
Training utilities for Gaussian Splatting.
Defines the Camera representation and a single differentiable training step.

Fixes vs original:
  1. nan_to_num_ on gradients BEFORE clip+step  -> breaks NaN into Adam m/v stats
  2. clip_grad_norm_ max_norm 1.0 -> 0.1 (conservative for MPS float32)
  3. mse/psnr inside torch.no_grad() -> no memory leak from dangling graph
  4. _clamp_params: nan_to_num_ + clamp on scales, quaternions, opacities
  5. opacity entropy regularization (faster convergence)
  6. debug=False default -> no per-step prints in production
"""

from dataclasses import dataclass
from typing import Optional

import torch

from torch_gs.renderer.renderer import render
from torch_gs.training.losses import l1_loss, d_ssim_loss


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """
    Pinhole camera for rendering.

    Attributes:
        W, H        : image resolution in pixels
        fx, fy      : focal lengths in pixel units
        cx, cy      : principal point coordinates
        W2C         : (4, 4) world-to-camera matrix  [torch.Tensor]
        full_proj   : (4, 4) full projection matrix  [torch.Tensor, optional]
    """
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    W2C: torch.Tensor
    full_proj: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GS_PARAM_NAMES = ("means", "scales", "quaternions", "opacities", "sh_coeffs")


def _gs_params(gaussians):
    return [getattr(gaussians, n) for n in _GS_PARAM_NAMES if hasattr(gaussians, n)]


def _sanitize_gradients(gaussians) -> None:
    """
    Zero out NaN/Inf gradients BEFORE optimizer.step().
    If the C++ rasterizer backward emits NaN, Adam accumulates them into
    its m/v statistics and every subsequent step stays NaN forever.
    """
    with torch.no_grad():
        for p in _gs_params(gaussians):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


def _clamp_params(gaussians) -> None:
    """
    Post-step parameter sanitization inside torch.no_grad().
    - log-scales clamped to [-7, 4]  ->  physical scale in [0.001, 54] units
    - quaternions re-normalized       ->  unit quaternion guarantee
    - NaN fallback for all params     ->  last-resort safety net
    """
    with torch.no_grad():
        gaussians.scales.nan_to_num_(nan=-3.0, posinf=2.0, neginf=-7.0)
        gaussians.scales.clamp_(-7.0, 0.5)

        gaussians.quaternions.nan_to_num_(nan=0.0)
        q = gaussians.quaternions
        norms = q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gaussians.quaternions.data = q / norms

        gaussians.opacities.nan_to_num_(nan=-2.0)
        gaussians.opacities.clamp_(-10.0, 6.0)


# ---------------------------------------------------------------------------
# Main training step
# ---------------------------------------------------------------------------

def train_step(
    gaussians,
    optimizer,
    target_image: torch.Tensor,
    camera: Camera,
    lambda_ssim: float = 0.2,
    device: str = "mps",
    rasterizer_type: str = "cpp",
    scale_reg_weight: float = 0.01,
    opacity_reg_weight: float = 0.001,
    debug: bool = False,
):
    """
    Single differentiable training step.

    Args:
        gaussians           : Gaussians object (.means/.scales/.quaternions/.opacities/.sh_coeffs)
        optimizer           : torch.optim.Optimizer over gaussians params
        target_image        : (H, W, 3) float32 in [0,1] on `device`
        camera              : Camera for this view
        lambda_ssim         : D-SSIM loss weight (default 0.2)
        device              : "mps" | "cuda" | "cpu"
        rasterizer_type     : "cpp" | "python"
        scale_reg_weight    : penalty on very small scales (prevents scale collapse)
        opacity_reg_weight  : binary entropy penalty (keeps opacities decisive)
        debug               : print loss breakdown each step (disable in production)

    Returns:
        (loss_val, psnr_val, rendered_image) — loss/psnr are Python floats
    """
    optimizer.zero_grad()

    # ---- Forward --------------------------------------------------------
    image = render(gaussians, camera, device=device, rasterizer_type=rasterizer_type)

    if debug:
        print(
            f"[LOSS DEBUG] image: nan={image.isnan().sum().item()} "
            f"range=[{image.min():.3f},{image.max():.3f}]",
            flush=True,
        )
        print(
            f"[LOSS DEBUG] target: nan={target_image.isnan().sum().item()} "
            f"range=[{target_image.min():.3f},{target_image.max():.3f}]",
            flush=True,
        )

    # ---- Loss -----------------------------------------------------------
    l1    = l1_loss(image, target_image)
    dssim = d_ssim_loss(image, target_image)
    loss  = (1.0 - lambda_ssim) * l1 + lambda_ssim * dssim

    if debug:
        print(f"[LOSS DEBUG] l1={l1.item():.5f}", flush=True)
        print(f"[LOSS DEBUG] d_ssim={dssim.item():.5f}", flush=True)

    # scale regularization: penalize tiny gaussians so they do not collapse to points
    if scale_reg_weight > 0.0:
        scale_mean = torch.exp(gaussians.scales).mean()
        loss = loss + scale_reg_weight * 0.1 * (1.0 / (scale_mean + 1e-6))

    # opacity entropy: pushes opacities toward 0 or 1, not stuck at 0.5
    if opacity_reg_weight > 0.0:
        sig = torch.sigmoid(gaussians.opacities)
        entropy = -(
            sig * torch.log(sig.clamp(min=1e-6))
            + (1.0 - sig) * torch.log((1.0 - sig).clamp(min=1e-6))
        ).mean()
        loss = loss + opacity_reg_weight * entropy

    # ---- Backward -------------------------------------------------------
    loss.backward()

    # FIX 1: zero NaN/Inf grads BEFORE Adam accumulates them into m/v stats
    _sanitize_gradients(gaussians)

    # FIX 2: conservative gradient clip (0.1 instead of 1.0)
    params_with_grad = [
        p for p in _gs_params(gaussians)
        if p.requires_grad and p.grad is not None
    ]
    if params_with_grad:
        torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=0.1)

    optimizer.step()

    # FIX 3: clamp/sanitize params post-step
    _clamp_params(gaussians)

    # FIX 4: metrics inside no_grad — image still has graph attached until here
    with torch.no_grad():
        mse  = ((image - target_image) ** 2).mean()
        psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))

    return loss.item(), psnr.item(), image.detach()
