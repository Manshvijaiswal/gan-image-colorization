"""
utils/color_utils.py
---------------------
Handles conversion between RGB and LAB color space, and normalization.

WHY THIS FILE MATTERS:
The model works entirely in LAB space. L is in [0, 100], a and b are
roughly in [-110, 110]. Neural networks train best when inputs are
roughly in [-1, 1], so we normalize before feeding to the network and
un-normalize before converting back to a viewable RGB image.
"""

import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb


def normalize_l(L):
    """L channel: [0, 100] -> [-1, 1]"""
    return (L / 50.0) - 1.0


def denormalize_l(L):
    """L channel: [-1, 1] -> [0, 100]"""
    return (L + 1.0) * 50.0


def normalize_ab(ab):
    """ab channels: [-110, 110] -> [-1, 1] (approx, safe clipping range)"""
    return ab / 110.0


def denormalize_ab(ab):
    """ab channels: [-1, 1] -> [-110, 110]"""
    return ab * 110.0


def rgb_to_lab_tensors(rgb_image):
    """
    rgb_image: numpy array, shape (H, W, 3), float in [0, 1]
    Returns:
        L  -> shape (1, H, W), normalized to [-1, 1]
        ab -> shape (2, H, W), normalized to [-1, 1]
    """
    lab = rgb2lab(rgb_image).astype("float32")   # (H, W, 3)
    L = lab[:, :, 0]
    ab = lab[:, :, 1:]

    L = normalize_l(L)
    ab = normalize_ab(ab)

    L = L[np.newaxis, :, :]                # (1, H, W)
    ab = np.transpose(ab, (2, 0, 1))        # (2, H, W)
    return L.astype("float32"), ab.astype("float32")


def lab_tensors_to_rgb(L, ab):
    """
    Inverse of rgb_to_lab_tensors, for visualization / saving output images.
    L  -> torch tensor or numpy, shape (1, H, W), normalized [-1, 1]
    ab -> torch tensor or numpy, shape (2, H, W), normalized [-1, 1]
    Returns: numpy RGB image (H, W, 3), float in [0, 1]

    NOTE: this uses skimage (CPU, non-differentiable) and is meant for
    saving/plotting single images (see utils/visualize.py). For anything
    used INSIDE the training loop (e.g. perceptual loss), use
    lab_batch_to_rgb_torch instead -- it stays on GPU and is differentiable.
    """
    if hasattr(L, "detach"):
        L = L.detach().cpu().numpy()
    if hasattr(ab, "detach"):
        ab = ab.detach().cpu().numpy()

    L = denormalize_l(L[0])                  # (H, W)
    ab = denormalize_ab(ab)                  # (2, H, W)
    ab = np.transpose(ab, (1, 2, 0))          # (H, W, 2)

    lab = np.concatenate([L[:, :, np.newaxis], ab], axis=2).astype("float64")
    rgb = lab2rgb(lab)
    return np.clip(rgb, 0, 1)


def lab_batch_to_rgb_torch(L_batch, ab_batch):
    """
    Fully vectorized, differentiable, GPU-resident LAB -> sRGB conversion.

    WHY THIS EXISTS (as opposed to reusing skimage's lab2rgb):
    skimage's lab2rgb only works on single numpy images on CPU and isn't
    differentiable. If used inside the perceptual-loss training step (once
    per batch, twice per batch counting real+fake), looping over each image
    and round-tripping GPU->CPU->GPU would badly bottleneck training speed
    and break gradient flow. This function implements the same LAB -> XYZ ->
    linear RGB -> sRGB math directly in torch tensor ops, so it runs on
    whatever device the input is on, processes the whole batch at once, and
    supports backpropagation (needed since perceptual loss must be able to
    push gradients back through the RGB reconstruction into the generator).

    L_batch:  (B, 1, H, W) normalized to [-1, 1]
    ab_batch: (B, 2, H, W) normalized to [-1, 1]
    Returns:  (B, 3, H, W) RGB tensor, values in [0, 1], same device/dtype
    """
    L = denormalize_l(L_batch)              # (B,1,H,W) -> [0, 100]
    ab = denormalize_ab(ab_batch)           # (B,2,H,W) -> [-110, 110]
    a = ab[:, 0:1, :, :]
    b = ab[:, 1:2, :, :]

    # ---- LAB -> XYZ ----
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def f_inv(t):
        return torch.where(t > delta, t ** 3, 3 * (delta ** 2) * (t - 4.0 / 29.0))

    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)

    # ---- XYZ -> linear sRGB ----
    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    Bc = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    rgb_linear = torch.cat([R, G, Bc], dim=1)
    rgb_linear = torch.clamp(rgb_linear, min=0.0)

    # ---- linear sRGB -> gamma-corrected sRGB ----
    threshold = 0.0031308
    rgb = torch.where(
        rgb_linear <= threshold,
        rgb_linear * 12.92,
        1.055 * torch.clamp(rgb_linear, min=threshold).pow(1 / 2.4) - 0.055,
    )
    return torch.clamp(rgb, 0.0, 1.0)
