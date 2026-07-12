"""
data/dataset.py
----------------
PyTorch Dataset that:
  1. Loads a color image from disk
  2. Resizes/augments it
  3. Converts RGB -> LAB
  4. Returns (L, ab) tensors: L is the model INPUT, ab is the TARGET

Any folder of color photos works as training data (see dataset section
in the guide for recommended sources: COCO, Places365, ImageNet subset).
"""

import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.color_utils import rgb_to_lab_tensors


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, image_size=256, split="train"):
        """
        root_dir: folder containing images (jpg/png), any subfolder depth
        split: "train" applies light augmentation, "val"/"test" does not
        """
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.paths = []
        for ext in exts:
            self.paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {root_dir}")

        self.split = split
        if split == "train":
            self.transform = T.Compose([
                T.Resize((image_size, image_size), Image.BICUBIC),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), Image.BICUBIC),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        img = np.array(img).astype("float32") / 255.0   # [0,1] RGB

        L, ab = rgb_to_lab_tensors(img)
        return torch.from_numpy(L), torch.from_numpy(ab)


if __name__ == "__main__":
    # quick sanity check (run: python -m data.dataset)
    ds = ColorizationDataset("data/raw/train", image_size=256)
    L, ab = ds[0]
    print("L shape:", L.shape, "ab shape:", ab.shape)
    print("L range:", L.min().item(), L.max().item())
    print("ab range:", ab.min().item(), ab.max().item())
