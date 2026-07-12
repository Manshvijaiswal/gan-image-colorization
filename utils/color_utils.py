"""
utils/color_utils.py
---------------------
Handles conversion between RGB and LAB color space, and normalization.
"""

import numpy as np
import torch


def normalize_l(L):
    return (L / 50.0) - 1.0


def denormalize_l(L):
    return (L + 1.0) * 50.0


def normalize_ab(ab):
    return ab / 110.0


def denormalize_ab(ab):
    return ab * 110.0


def rgb_to_lab_tensors(rgb_image):
    from skimage.color import rgb2lab
    lab = rgb2lab(rgb_image).astype("float32")
    L = lab[:, :, 0]
    ab = lab[:, :, 1:]
    L = normalize_l(L)
    ab = normalize_ab(ab)
    L = L[np.newaxis, :, :]
    ab = np.transpose(ab, (2, 0, 1))
    return L.astype("float32"), ab.astype("float32")


def lab_tensors_to_rgb(L, ab):
    from skimage.color import lab2rgb
    if hasattr(L, "detach"):
        L = L.detach().cpu().numpy()
    if hasattr(ab, "detach"):
        ab = ab.detach().cpu().numpy()
    L = denormalize_l(L[0])
    ab = denormalize_ab(ab)
    ab = np.transpose(ab, (1, 2, 0))
    lab = np.concatenate([L[:, :, np.newaxis], ab], axis=2).astype("float64")
    rgb = lab2rgb(lab)
    return np.clip(rgb, 0, 1)


def lab_batch_to_rgb_torch(L_batch, ab_batch):
    L = denormalize_l(L_batch)
    ab = denormalize_ab(ab_batch)
    a = ab[:, 0:1, :, :]
    b = ab[:, 1:2, :, :]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def f_inv(t):
        return torch.where(t > delta, t ** 3, 3 * (delta ** 2) * (t - 4.0 / 29.0))

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)

    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    Bc = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    rgb_linear = torch.cat([R, G, Bc], dim=1)
    rgb_linear = torch.clamp(rgb_linear, min=0.0)

    threshold = 0.0031308
    rgb = torch.where(
        rgb_linear <= threshold,
        rgb_linear * 12.92,
        1.055 * torch.clamp(rgb_linear, min=threshold).pow(1 / 2.4) - 0.055,
    )
    return torch.clamp(rgb, 0.0, 1.0)
