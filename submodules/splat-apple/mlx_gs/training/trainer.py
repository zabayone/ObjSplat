import mlx.core as mx
from mlx_gs.renderer.renderer import render
from mlx_gs.training.losses import l1_loss, d_ssim_loss, blur_aware_loss
from dataclasses import dataclass

@dataclass
class Camera:
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    W2C: mx.array
    full_proj: mx.array = None


def _to_luma(img):
    if img.shape[-1] == 1:
        return img
    weights = mx.array([0.299, 0.587, 0.114], dtype=img.dtype)
    return mx.sum(img * weights[None, None, None, :], axis=-1, keepdims=True)


def _align_image_pair(img1, img2):
    if len(img1.shape) == 3:
        img1 = img1[None, ...]
    if len(img2.shape) == 3:
        img2 = img2[None, ...]

    h = min(int(img1.shape[1]), int(img2.shape[1]))
    w = min(int(img1.shape[2]), int(img2.shape[2]))

    def crop(img):
        top = max((int(img.shape[1]) - h) // 2, 0)
        left = max((int(img.shape[2]) - w) // 2, 0)
        return img[:, top:top + h, left:left + w, :]

    img1 = crop(img1)
    img2 = crop(img2)

    if img1.shape[-1] != img2.shape[-1]:
        if img1.shape[-1] == 1:
            img2 = _to_luma(img2)
        elif img2.shape[-1] == 1:
            img1 = _to_luma(img1)
        else:
            img1 = _to_luma(img1)
            img2 = _to_luma(img2)

    return img1, img2


def _align_mask(mask, reference):
    if mask is None:
        return None
    if len(mask.shape) == 2:
        mask = mask[None, ..., None]
    elif len(mask.shape) == 3:
        mask = mask[None, ...]
    if len(reference.shape) == 3:
        reference = reference[None, ...]
    h = min(int(mask.shape[1]), int(reference.shape[1]))
    w = min(int(mask.shape[2]), int(reference.shape[2]))
    top = max((int(mask.shape[1]) - h) // 2, 0)
    left = max((int(mask.shape[2]) - w) // 2, 0)
    return mx.clip(mask[:, top:top + h, left:left + w, :1], 0.0, 1.0)


def loss_fn(
    params,
    target_image,
    camera,
    lambda_ssim,
    rasterizer_type,
    active_sh_degree: int = 1,
    target_mask=None,
):
    """
    Computes loss for MLX value_and_grad.
    params is a dict or dataclass of MLX arrays.
    """
    image = render(params, camera, rasterizer_type=rasterizer_type, active_sh_degree=active_sh_degree)

    image, target_image = _align_image_pair(image, target_image)

    target_mask = _align_mask(target_mask, image)
    if target_mask is None:
        l1 = l1_loss(image, target_image)
        d_ssim = d_ssim_loss(image, target_image)
    else:
        # Pixels outside a layer are unknown, not black. Mask both inputs
        # before the structural term and normalize the pixel term only over
        # supervised pixels, preventing dark halos at layer boundaries.
        denom = mx.maximum(mx.sum(target_mask) * image.shape[-1], 1.0)
        l1 = mx.sum(mx.abs(image - target_image) * target_mask) / denom
        d_ssim = d_ssim_loss(image * target_mask, target_image * target_mask)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim
    
    return loss, image


def _repulsion_loss(means, max_samples: int = 512, min_dist: float = 0.02):
    """Encourage gaussian centers to stay separated without an O(N^2) global penalty."""

    n = means.shape[0]
    if n < 2:
        return mx.array(0.0, dtype=mx.float32)

    sample_count = int(min(max_samples, n))
    if sample_count < 2:
        return mx.array(0.0, dtype=mx.float32)

    indices = mx.random.permutation(n)[:sample_count]
    sampled = means[indices]
    deltas = sampled[:, None, :] - sampled[None, :, :]
    distances = mx.sqrt(mx.sum(mx.square(deltas), axis=-1) + 1e-12)
    mask = 1.0 - mx.eye(sample_count, dtype=mx.float32)
    hinge = mx.maximum(0.0, min_dist - distances)
    return mx.sum(hinge * hinge * mask) / mx.maximum(mx.sum(mask), 1.0)


def _clamp_params(params, scale_log_min: float = -7.0, scale_log_max: float = 0.1):
    """Sanitize MLX gaussian parameters after each optimizer step."""

    try:
        scale_arr = params["scales"] if isinstance(params, dict) else params.scales
        scale_arr = mx.nan_to_num(scale_arr, nan=-3.0, posinf=2.0, neginf=-7.0)
        scale_arr = mx.clip(scale_arr, scale_log_min, scale_log_max)
        if isinstance(params, dict):
            params["scales"] = scale_arr
        else:
            params.scales = scale_arr

        quat_arr = params["quaternions"] if isinstance(params, dict) else params.quaternions
        quat_arr = mx.nan_to_num(quat_arr, nan=0.0)
        quat_norm = mx.sqrt(mx.sum(mx.square(quat_arr), axis=-1, keepdims=True) + 1e-12)
        quat_arr = quat_arr / quat_norm
        if isinstance(params, dict):
            params["quaternions"] = quat_arr
        else:
            params.quaternions = quat_arr

        op_arr = params["opacities"] if isinstance(params, dict) else params.opacities
        op_arr = mx.nan_to_num(op_arr, nan=-2.0, posinf=6.0, neginf=-10.0)
        op_arr = mx.clip(op_arr, -4.0, 6.0)
        if isinstance(params, dict):
            params["opacities"] = op_arr
        else:
            params.opacities = op_arr
    except Exception:
        pass

def train_step(
    params,
    optimizers,
    target_image,
    camera,
    lambda_ssim=0.2,
    rasterizer_type="cpp",
    opacity_reg_weight: float = 0.001,
    opacity_mean_reg_weight: float = 0.0,
    scale_reg_weight: float = 0.0,
    blur_reg_weight: float = 0.0,
    scale_log_min: float = -7.0,
    scale_log_max: float = 0.1,
    repulsion_weight: float = 0.0,
    repulsion_min_dist: float = 0.02,
    repulsion_max_samples: int = 512,
    active_sh_degree: int = 1,
    target_mask=None,
):
    def wrapped_loss(p):
        loss, image = loss_fn(
            p,
            target_image,
            camera,
            lambda_ssim,
            rasterizer_type,
            active_sh_degree=active_sh_degree,
            target_mask=target_mask,
        )
        image, aligned_target_image = _align_image_pair(image, target_image)
        aligned_target_mask = _align_mask(target_mask, image)

        if opacity_reg_weight > 0.0:
            op = p["opacities"] if isinstance(p, dict) else p.opacities
            sig = mx.sigmoid(op)
            entropy = -(
                sig * mx.log(sig + 1e-8)
                + (1.0 - sig) * mx.log(1.0 - sig + 1e-8)
            )
            loss = loss + opacity_reg_weight * mx.mean(entropy)

        if opacity_mean_reg_weight > 0.0:
            op = p["opacities"] if isinstance(p, dict) else p.opacities
            sig = mx.sigmoid(op)
            loss = loss + opacity_mean_reg_weight * mx.mean(sig)

        if scale_reg_weight > 0.0:
            scales = p["scales"] if isinstance(p, dict) else p.scales
            scale_mean = mx.mean(mx.exp(scales))
            loss = loss + scale_reg_weight * 0.1 * (1.0 / (scale_mean + 1e-6))

        if blur_reg_weight > 0.0:
            if aligned_target_mask is None:
                loss = loss + blur_reg_weight * blur_aware_loss(
                    image, aligned_target_image
                )
            else:
                loss = loss + blur_reg_weight * blur_aware_loss(
                    image * aligned_target_mask,
                    aligned_target_image * aligned_target_mask,
                )

        if repulsion_weight > 0.0:
            means = p["means"] if isinstance(p, dict) else p.means
            loss = loss + repulsion_weight * _repulsion_loss(
                means,
                max_samples=repulsion_max_samples,
                min_dist=repulsion_min_dist,
            )

        return loss, image

    loss_and_grad_fn = mx.value_and_grad(wrapped_loss)
    (loss, rendered_image), grads = loss_and_grad_fn(params)

    if isinstance(optimizers, dict):
        for key in grads:
            if key in optimizers:
                params[key] = optimizers[key].apply_gradients({key: grads[key]}, {key: params[key]})[key]
    else:
        optimizers.update(params, grads)

    _clamp_params(params, scale_log_min, scale_log_max)

    rendered_image, target_image = _align_image_pair(rendered_image, target_image)
    target_mask = _align_mask(target_mask, rendered_image)
    if target_mask is None:
        mse = mx.mean(mx.square(rendered_image - target_image))
    else:
        denom = mx.maximum(mx.sum(target_mask) * rendered_image.shape[-1], 1.0)
        mse = mx.sum(mx.square(rendered_image - target_image) * target_mask) / denom
    psnr = -10.0 * mx.log10(mx.maximum(mse, 1e-10))

    return loss, rendered_image, psnr, {}  # grad_norms vuoto, non sync
