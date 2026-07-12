"""
inference.py
--------------
Load a trained generator and colorize a single grayscale image, or a
whole folder of images. This is what the Streamlit/Flask app calls
under the hood.

Usage:
    python inference.py --checkpoint checkpoints/generator_final.pth \
                         --input path/to/grayscale.jpg \
                         --output path/to/colorized.jpg
"""

import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from config import cfg
from models.generator import UNetGenerator
from utils.color_utils import normalize_l, lab_tensors_to_rgb


def load_generator(checkpoint_path, device=None):
    device = device or cfg.DEVICE
    G = UNetGenerator(in_channels=1, out_channels=2, features=cfg.GEN_FEATURES).to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()
    return G


def colorize_image_from_pil(G, pil_image, image_size=256, device=None):
    """
    Same logic as colorize_image(), but takes an already-loaded PIL image
    instead of a file path. Used by the Flask API (web/server.py), where
    the image arrives as an in-memory upload rather than a file on disk.
    """
    device = device or cfg.DEVICE

    img = pil_image.convert("L")
    original_size = img.size
    img_resized = img.resize((image_size, image_size), Image.BICUBIC)

    l_arr = np.array(img_resized).astype("float32") / 255.0 * 100.0
    l_norm = normalize_l(l_arr)
    L_tensor = torch.from_numpy(l_norm[np.newaxis, np.newaxis, :, :]).float().to(device)

    with torch.no_grad():
        fake_ab = G(L_tensor)

    rgb = lab_tensors_to_rgb(L_tensor[0].cpu(), fake_ab[0].cpu())
    out_img = Image.fromarray((rgb * 255).astype(np.uint8))
    out_img = out_img.resize(original_size, Image.BICUBIC)
    return out_img


def colorize_image(G, image_path, image_size=256, device=None):
    """
    Takes ANY input image (grayscale or color -- if color, we discard
    the color and only use luminance, so a user can test on already
    colored images too). Returns a PIL RGB image.
    """
    device = device or cfg.DEVICE
    img = Image.open(image_path)
    return colorize_image_from_pil(G, img, image_size=image_size, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="colorized_output.jpg")
    args = parser.parse_args()

    G = load_generator(args.checkpoint)
    result = colorize_image(G, args.input)
    result.save(args.output)
    print(f"Saved colorized image to {args.output}")
