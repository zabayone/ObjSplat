import mlx.core as mx
from dataclasses import dataclass
import numpy as np


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))

@dataclass
class Gaussians:
    """
    Main data structure representing a collection of 3D Gaussians in MLX.
    
    Attributes:
        means: (N, 3) array of central positions.
        scales: (N, 3) array of log-scales.
        quaternions: (N, 4) array of orientations (w, x, y, z).
        opacities: (N, 1) array of raw opacities.
        sh_coeffs: (N, K, 3) array of SH coefficients.
    """
    means: mx.array
    scales: mx.array
    quaternions: mx.array
    opacities: mx.array
    sh_coeffs: mx.array

def init_gaussians_from_pcd(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    sh_degree: int = 3,
    scale_init: np.ndarray | float | None = None,
    opacity_init: float = 0.1,
) -> Gaussians:
    """
    Initializes MLX-based Gaussians from a point cloud.
    """
    means = mx.array(points, dtype=mx.float32)
    
    # Initialize scales (log space)
    if scale_init is None:
        scales = mx.full((points.shape[0], 3), -5.0, dtype=mx.float32)
    else:
        scale_init = np.asarray(scale_init, dtype=np.float32)
        if scale_init.ndim == 0:
            scale_init = np.full((points.shape[0], 3), float(scale_init), dtype=np.float32)
        elif scale_init.ndim == 1:
            scale_init = np.repeat(scale_init[:, None], 3, axis=1)
        scale_init = scale_init + np.random.normal(0.0, 0.02, size=scale_init.shape).astype(np.float32)
        scales = mx.array(scale_init, dtype=mx.float32)
    
    # Initialize quaternions near identity but with a small random perturbation
    # so the optimizer can break symmetry and specialize axes earlier.
    w = mx.ones((points.shape[0], 1), dtype=mx.float32)
    xyz_q = mx.array(
        np.random.normal(0.0, 0.1, size=(points.shape[0], 3)).astype(np.float32)
    )
    quaternions = mx.concatenate([w, xyz_q], axis=1)
    quat_norm = mx.sqrt(mx.sum(mx.square(quaternions), axis=-1, keepdims=True) + 1e-12)
    quaternions = quaternions / quat_norm
    
    # Initialize opacities (raw)
    opacities = mx.full((points.shape[0], 1), _logit(opacity_init), dtype=mx.float32)
    
    # SH Coefficients (DC term + rest zeros for LayerPano compatibility)
    # SH DC = (color - 0.5) / 0.28209479177387814
    sh_dc = (mx.array(colors, dtype=mx.float32) - 0.5) / 0.28209479177387814
    sh_dc = sh_dc[:, None, :]
    sh_rest = mx.zeros((points.shape[0], (sh_degree + 1) ** 2 - 1, 3), dtype=mx.float32)
    sh_coeffs = mx.concatenate([sh_dc, sh_rest], axis=1)
    
    return Gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        sh_coeffs=sh_coeffs
    )

def get_covariance_3d(scales: mx.array, quaternions: mx.array) -> mx.array:
    """
    Computes 3D covariance matrix Sigma = R S S^T R^T.
    """
    # 1. Scaling matrix S
    s = mx.exp(scales)
    
    # Stable normalization
    norm = mx.sqrt(mx.sum(mx.square(quaternions), axis=-1, keepdims=True) + 1e-12)
    q = quaternions / norm
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Build rotation matrix rows (MLX style broadcasting)
    # R = mx.zeros((q.shape[0], 3, 3))
    # R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    # ...
    
    r00 = 1 - 2 * (y*y + z*z)
    r01 = 2 * (x*y - r*z)
    r02 = 2 * (x*z + r*y)
    
    r10 = 2 * (x*y + r*z)
    r11 = 1 - 2 * (x*x + z*z)
    r12 = 2 * (y*z - r*x)
    
    r20 = 2 * (x*z - r*y)
    r21 = 2 * (y*z + r*x)
    r22 = 1 - 2 * (x*x + y*y)
    
    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    
    R = mx.stack([row0, row1, row2], axis=-2)
    
    # 3. M = R * S
    # S is diagonal, so we can just scale columns of R
    M = R * s[:, None, :]
    
    # 4. Sigma = M * M^T
    Sigma = mx.matmul(M, M.transpose(0, 2, 1))
    
    return Sigma
