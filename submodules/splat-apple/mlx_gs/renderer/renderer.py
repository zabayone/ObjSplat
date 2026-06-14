import mlx.core as mx
from mlx_gs.renderer.projection import project_gaussians
from mlx_gs.renderer import rasterizer
try:
    from mlx_gs.renderer import rasterizer_metal
except ImportError:
    rasterizer_metal = None

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


def _eval_sh_rgb(sh_coeffs, view_dirs, active_sh_degree: int = 1):
    rgb = sh_coeffs[:, 0, :] * SH_C0
    if active_sh_degree < 1 or sh_coeffs.shape[1] < 4:
        return mx.clip(rgb + 0.5, 0.0, 1.0)

    x = view_dirs[:, 0][:, None]
    y = view_dirs[:, 1][:, None]
    z = view_dirs[:, 2][:, None]
    rgb = rgb + SH_C1 * (
        (-y) * sh_coeffs[:, 1, :]
        + z * sh_coeffs[:, 2, :]
        + (-x) * sh_coeffs[:, 3, :]
    )
    return mx.clip(rgb + 0.5, 0.0, 1.0)


def _render_stage1(params, camera_dict, active_sh_degree: int = 1):
    # Standard projection
    class Obj:
        def __init__(self, d): self.__dict__.update(d)
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(Obj(params), Obj(camera_dict))

    means3D = params["means"]
    means_homo = mx.concatenate([means3D, mx.ones((means3D.shape[0], 1), dtype=means3D.dtype)], axis=1)
    means_cam = (camera_dict["W2C"] @ means_homo.T).T
    view_norm = mx.sqrt(mx.sum(mx.square(means_cam), axis=-1, keepdims=True) + 1e-8)
    view_dirs = means_cam / view_norm

    colors = _eval_sh_rgb(params["sh_coeffs"], view_dirs, active_sh_degree=active_sh_degree)
    
    return means2D, cov2D, radii, valid_mask, depths, colors

def render(gaussians, camera, background=None, rasterizer_type="python", active_sh_degree: int = 1):
    if isinstance(gaussians, dict):
        params = gaussians
    else:
        params = {
            "means": gaussians.means, "scales": gaussians.scales,
            "quaternions": gaussians.quaternions, "opacities": gaussians.opacities,
            "sh_coeffs": gaussians.sh_coeffs
        }
    
    cam_dict = {
        "H": camera.H, "W": camera.W, "fx": camera.fx, "fy": camera.fy,
        "cx": camera.cx, "cy": camera.cy, "W2C": camera.W2C
    }
    
    # Stage 1: Projection (Compiled independent of interactions)
    means2D, cov2D, radii, valid_mask, depths, colors = _render_stage1(
        params, cam_dict, active_sh_degree=active_sh_degree
    )
    
    if rasterizer_type == "cpp":
        if rasterizer_metal is None or rasterizer_metal.rasterizer_metal is None:
            raise ImportError("Metal rasterizer not available. Please build the extension.")
            
        # Phase 4: Use GPU-resident interactions
        sorted_tile_ids, sorted_gaussian_ids = rasterizer_metal.get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, rasterizer.TILE_SIZE
        )
        
        return rasterizer_metal.render_tiles(
            means2D, cov2D, params["opacities"], colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, rasterizer.TILE_SIZE, background
        )
    else:
        # Python Interaction Stage
        sorted_tile_ids, sorted_gaussian_ids = rasterizer._get_tile_interactions_impl(
            means2D, radii, valid_mask, depths, camera.H, camera.W, rasterizer.TILE_SIZE
        )
        return rasterizer.render_tiles(
            means2D, cov2D, params["opacities"], colors, 
            sorted_tile_ids, sorted_gaussian_ids, 
            camera.H, camera.W, rasterizer.TILE_SIZE, background
        )
