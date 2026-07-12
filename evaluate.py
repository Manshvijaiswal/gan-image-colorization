"""
evaluate.py
------------
Runs the trained generator over a test set and reports PSNR, SSIM, and
(optionally) FID. Also saves generated images to disk so FID can be
computed against the real folder.

Usage:
    python evaluate.py --checkpoint checkpoints/generator_final.pth
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from config import cfg
from data.dataset import ColorizationDataset
from models.generator import UNetGenerator
from utils.color_utils import lab_tensors_to_rgb
from utils.metrics import compute_batch_metrics, compute_fid


def evaluate(checkpoint_path, test_dir, save_generated_dir="outputs/test_generated"):
    os.makedirs(save_generated_dir, exist_ok=True)
    real_dir_for_fid = "outputs/test_real"
    os.makedirs(real_dir_for_fid, exist_ok=True)

    G = UNetGenerator(in_channels=1, out_channels=2, features=cfg.GEN_FEATURES).to(cfg.DEVICE)
    G.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
    G.eval()

    test_ds = ColorizationDataset(test_dir, image_size=cfg.IMAGE_SIZE, split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    real_rgb_list, fake_rgb_list = [], []
    img_counter = 0

    with torch.no_grad():
        for L, real_ab in test_loader:
            L = L.to(cfg.DEVICE)
            fake_ab = G(L)

            for i in range(L.shape[0]):
                real_rgb = lab_tensors_to_rgb(L[i].cpu(), real_ab[i])
                fake_rgb = lab_tensors_to_rgb(L[i].cpu(), fake_ab[i].cpu())

                real_rgb_list.append(real_rgb)
                fake_rgb_list.append(fake_rgb)

                Image.fromarray((real_rgb * 255).astype(np.uint8)).save(
                    os.path.join(real_dir_for_fid, f"{img_counter:05d}.png"))
                Image.fromarray((fake_rgb * 255).astype(np.uint8)).save(
                    os.path.join(save_generated_dir, f"{img_counter:05d}.png"))
                img_counter += 1

    metrics = compute_batch_metrics(real_rgb_list, fake_rgb_list)
    print(f"PSNR: {metrics['PSNR_mean']:.2f} dB")
    print(f"SSIM: {metrics['SSIM_mean']:.4f}")

    try:
        fid_value = compute_fid(real_dir_for_fid, save_generated_dir)
        print(f"FID: {fid_value:.2f}")
        metrics["FID"] = fid_value
    except ImportError:
        print("pytorch-fid not installed -- skipping FID. Run: pip install pytorch-fid")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=cfg.VAL_DIR)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.test_dir)
