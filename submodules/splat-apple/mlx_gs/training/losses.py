import mlx.core as mx
import math


def _to_batched(img):
    if len(img.shape) == 3:
        return img[None, ...]
    return img


def _to_luma(img):
    if img.shape[-1] == 1:
        return img
    weights = mx.array([0.299, 0.587, 0.114], dtype=img.dtype)
    return mx.sum(img * weights[None, None, None, :], axis=-1, keepdims=True)


def _align_pair(img1, img2):
    img1 = _to_batched(img1)
    img2 = _to_batched(img2)

    h = min(int(img1.shape[1]), int(img2.shape[1]))
    w = min(int(img1.shape[2]), int(img2.shape[2]))

    def crop(img):
        top = max((int(img.shape[1]) - h) // 2, 0)
        left = max((int(img.shape[2]) - w) // 2, 0)
        return img[:, top:top + h, left:left + w, :]

    img1 = crop(img1)
    img2 = crop(img2)

    if img1.shape[-1] != img2.shape[-1]:
        img1 = _to_luma(img1)
        img2 = _to_luma(img2)

    return img1, img2

def l1_loss(output, gt):
    output, gt = _align_pair(output, gt)
    return mx.mean(mx.abs(output - gt))

def l2_loss(output, gt):
    output, gt = _align_pair(output, gt)
    return mx.mean(mx.square(output - gt))

def gaussian_kernel(window_size, sigma):
    x = mx.arange(window_size) - window_size // 2
    gauss = mx.exp(-mx.square(x) / (2 * sigma**2))
    return gauss / mx.sum(gauss)

def create_window(window_size, channel):
    _1d = gaussian_kernel(window_size, 1.5)
    _2d = mx.outer(_1d, _1d)
    # For depthwise convolution in MLX (groups=C):
    # Weight shape should be (C, window_size, window_size, 1)
    return mx.tile(_2d[None, :, :, None], (channel, 1, 1, 1))


def create_window_cached(window_size, channel):
    # MLX version keeps the API expected by ssim() while avoiding a hard NameError.
    return create_window(window_size, channel)

def ssim(img1, img2, window_size=11):
    img1, img2 = _align_pair(img1, img2)
    C = img1.shape[-1]
    window = create_window_cached(window_size, C)
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Use depthwise convolution (groups=C)
    mu1 = mx.conv2d(img1, window, stride=1, padding=window_size//2, groups=C)
    mu2 = mx.conv2d(img2, window, stride=1, padding=window_size//2, groups=C)
    
    mu1_sq = mx.square(mu1)
    mu2_sq = mx.square(mu2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = mx.conv2d(mx.square(img1), window, stride=1, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = mx.conv2d(mx.square(img2), window, stride=1, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = mx.conv2d(img1 * img2, window, stride=1, padding=window_size//2, groups=C) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return mx.mean(ssim_map)

def d_ssim_loss(img1, img2):
    """
    1 - SSIM loss in MLX.
    Ensures BHWC format.
    """
    img1, img2 = _align_pair(img1, img2)
    return 1.0 - ssim(img1, img2)


def _center_crop_to_match(img1, img2):
    return _align_pair(img1, img2)

def blur_aware_loss(img1, img2):
    """
    Penalize blur by matching Laplacian edge responses between render and target.
    Operates on luminance so it is less sensitive to color noise.
    """
    img1, img2 = _center_crop_to_match(img1, img2)
    gray1 = _to_luma(img1)
    gray2 = _to_luma(img2)

    if gray1.shape[1] < 3 or gray1.shape[2] < 3:
        return mx.array(0.0, dtype=img1.dtype)

    center1 = gray1[:, 1:-1, 1:-1, :]
    center2 = gray2[:, 1:-1, 1:-1, :]

    lap1 = (
        -4.0 * center1
        + gray1[:, :-2, 1:-1, :]
        + gray1[:, 2:, 1:-1, :]
        + gray1[:, 1:-1, :-2, :]
        + gray1[:, 1:-1, 2:, :]
    )
    lap2 = (
        -4.0 * center2
        + gray2[:, :-2, 1:-1, :]
        + gray2[:, 2:, 1:-1, :]
        + gray2[:, 1:-1, :-2, :]
        + gray2[:, 1:-1, 2:, :]
    )

    return mx.mean(mx.abs(lap1 - lap2))
