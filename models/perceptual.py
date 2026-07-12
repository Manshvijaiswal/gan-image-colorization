"""
models/perceptual.py
----------------------
Perceptual (VGG) Loss -- an ADVANCED IMPROVEMENT beyond basic pix2pix.

CONCEPT:
Instead of comparing raw pixels between fake and real images, we pass
both through a pretrained VGG16 (trained on ImageNet) and compare their
INTERMEDIATE FEATURE MAPS. These feature maps encode texture and semantic
content (edges, patterns, object parts) rather than exact pixel values.

WHY THIS HELPS COLORIZATION:
Pixel-only losses (L1/L2) can be satisfied by "safe" desaturated colors.
Perceptual loss instead asks "does this look texturally/semantically
similar to a real photo of this kind of scene?" -- pushing the generator
toward outputs that are semantically consistent (e.g. foliage looks
textured and green-ish, sky looks smooth and sky-like) rather than merely
pixel-close.

NOTE: VGG expects standard RGB input, so we reconstruct an RGB image
from (L, ab) before passing it through VGG.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(3, 8, 15, 22)):
        """
        layers: indices into vgg16.features at which to extract activations
                (roughly: relu1_2, relu2_2, relu3_3, relu4_3)
        """
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False   # freeze VGG -- we only use it to extract features

        self.vgg = vgg
        self.layers = set(layers)
        self.criterion = nn.L1Loss()

        # ImageNet normalization stats (VGG was trained with these)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                feats.append(x)
        return feats

    def forward(self, fake_rgb, real_rgb):
        """
        fake_rgb, real_rgb: (B, 3, H, W), values in [0, 1]
        """
        fake_feats = self._extract(fake_rgb)
        real_feats = self._extract(real_rgb)
        loss = 0.0
        for f, r in zip(fake_feats, real_feats):
            loss = loss + self.criterion(f, r)
        return loss
