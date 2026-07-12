"""
utils/visualize.py
--------------------
Helpers to save side-by-side comparison grids: grayscale input |
ground truth color | generated color. Used during training to sanity
check progress, and during inference/reporting.
"""

import matplotlib
matplotlib.use("Agg")   # no display needed, just save to file
import matplotlib.pyplot as plt
import numpy as np

from utils.color_utils import lab_tensors_to_rgb


def save_sample_grid(L_batch, real_ab_batch, fake_ab_batch, save_path, max_images=4):
    """
    L_batch, real_ab_batch, fake_ab_batch: torch tensors, batch dimension first,
    already on CPU. Saves a grid: rows = samples, cols = [gray | real | fake].
    """
    n = min(max_images, L_batch.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        L = L_batch[i]
        real_ab = real_ab_batch[i]
        fake_ab = fake_ab_batch[i]

        gray = (L[0].numpy() + 1) / 2.0  # back to [0,1] for display
        real_rgb = lab_tensors_to_rgb(L, real_ab)
        fake_rgb = lab_tensors_to_rgb(L, fake_ab)

        axes[i, 0].imshow(gray, cmap="gray")
        axes[i, 0].set_title("Grayscale Input")
        axes[i, 1].imshow(real_rgb)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(fake_rgb)
        axes[i, 2].set_title("Generated")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
