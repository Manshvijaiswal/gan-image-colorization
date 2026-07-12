"""
utils/metrics.py
-------------------
Evaluation metrics for colorization quality.

WHAT EACH METRIC TELLS YOU:

- PSNR (Peak Signal-to-Noise Ratio): pixel-level fidelity. Higher = closer
  to ground truth pixel values. BUT: a "safe," desaturated colorization can
  score deceptively high PSNR while looking dull -- so never report PSNR
  alone.

- SSIM (Structural Similarity Index): compares luminance, contrast, and
  structure between two images. More aligned with human perception of
  structural similarity than raw PSNR. Range: [-1, 1], higher is better.

- FID (Frechet Inception Distance): compares the DISTRIBUTION of generated
  images to the distribution of real images, using features from a
  pretrained Inception network. This captures realism and diversity, not
  just per-pixel closeness. Lower FID = the generated images' feature
  distribution is closer to that of real images. This is the metric most
  correlated with human "does this look real" judgment, and is standard in
  GAN literature.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_psnr(real_rgb, fake_rgb):
    """real_rgb, fake_rgb: numpy arrays (H, W, 3), float in [0, 1]"""
    return psnr(real_rgb, fake_rgb, data_range=1.0)


def compute_ssim(real_rgb, fake_rgb):
    """real_rgb, fake_rgb: numpy arrays (H, W, 3), float in [0, 1]"""
    return ssim(real_rgb, fake_rgb, data_range=1.0, channel_axis=2)


def compute_batch_metrics(real_rgb_list, fake_rgb_list):
    """Average PSNR/SSIM across a list of image pairs."""
    psnrs, ssims = [], []
    for real, fake in zip(real_rgb_list, fake_rgb_list):
        psnrs.append(compute_psnr(real, fake))
        ssims.append(compute_ssim(real, fake))
    return {
        "PSNR_mean": float(np.mean(psnrs)),
        "SSIM_mean": float(np.mean(ssims)),
    }


def compute_fid(real_dir, fake_dir):
    """
    Computes FID between two folders of images using the `pytorch-fid`
    package (industry-standard implementation).

    Install:  pip install pytorch-fid
    Usage:    compute_fid("data/real_test_images", "outputs/generated_images")

    We shell out to pytorch-fid's CLI-equivalent API rather than
    reimplementing Inception feature extraction ourselves -- this ensures
    results are comparable to published papers, which all use this exact
    implementation.
    """
    from pytorch_fid.fid_score import calculate_fid_given_paths
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_value = calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=50,
        device=device,
        dims=2048,   # standard Inception pool3 feature dimension
    )
    return fid_value
