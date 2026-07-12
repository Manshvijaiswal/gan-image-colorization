"""
models/discriminator.py
-------------------------
PatchGAN Discriminator.

WHY PATCHGAN INSTEAD OF A NORMAL CLASSIFIER:
A normal discriminator outputs ONE number for the whole image: real or
fake. That's a very coarse learning signal and tends to produce blurry,
low-detail generators. PatchGAN instead outputs an NxN grid, where each
value corresponds to whether a *local patch* (e.g. 70x70 receptive field)
of the image looks real. This:
  - gives many localized gradients per image (richer training signal)
  - directly penalizes local color blotches / artifacts
  - has fewer parameters (fully convolutional, no dense layers)

INPUT: the discriminator is CONDITIONAL -- it sees both the L channel
(condition) and the ab channels (real or generated) concatenated together,
so it can judge "does this color make sense given this grayscale image?"
not just "does this look like a plausible color image in general?"

INPUT:  (B, 3, H, W)  -- L + ab concatenated (1 + 2 = 3 channels)
OUTPUT: (B, 1, 30, 30) -- grid of real/fake scores (for 256x256 input)
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        def block(in_ch, out_ch, stride=2, use_norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                                 padding=1, bias=not use_norm)]
            if use_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, features, use_norm=False),   # 256 -> 128
            *block(features, features * 2),                  # 128 -> 64
            *block(features * 2, features * 4),               # 64  -> 32
            *block(features * 4, features * 8, stride=1),      # 32  -> 31 (stride 1 keeps receptive field growing without shrinking too much)
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)  # -> 30x30x1 patch map
        )

    def forward(self, L, ab):
        x = torch.cat([L, ab], dim=1)   # condition on L (the grayscale input)
        return self.model(x)


if __name__ == "__main__":
    model = PatchDiscriminator()
    L = torch.randn(2, 1, 256, 256)
    ab = torch.randn(2, 2, 256, 256)
    out = model(L, ab)
    print("Output shape:", out.shape)   # expect (2, 1, 30, 30)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
