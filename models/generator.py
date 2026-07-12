"""
models/generator.py
--------------------
U-Net Generator: takes the L (lightness) channel and predicts the a,b
color channels.

WHY U-NET:
Standard encoder-decoders squeeze the whole image into a small bottleneck,
losing fine spatial detail (edges, textures). U-Net adds "skip connections"
that copy feature maps directly from encoder layer i to decoder layer
(n-i), so fine details bypass the bottleneck entirely. This is crucial for
colorization: color boundaries must align exactly with object edges.

INPUT:  (B, 1, H, W)   L channel
OUTPUT: (B, 2, H, W)   predicted a,b channels (values in [-1, 1] via Tanh)
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """One encoder block: Conv -> (optional BatchNorm) -> LeakyReLU"""
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    """
    One decoder block: Upsample (nearest) -> Conv -> BatchNorm -> ReLU -> (optional Dropout)

    NOTE: We deliberately use Upsample+Conv instead of ConvTranspose2d here.
    ConvTranspose2d with kernel_size=4, stride=2 creates uneven kernel overlap
    across the output, producing regular "checkerboard"/polka-dot artifacts --
    especially visible in smooth regions like sky/water once adversarial
    training starts exploiting the pattern. Upsample+Conv avoids this because
    the upsampling step is a fixed, artifact-free operation, and the conv
    that follows only smooths/blends -- it can't introduce grid artifacts.
    """
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.block(x)
        # concatenate along channel dimension -> the "skip connection"
        return torch.cat([x, skip_input], dim=1)


class UNetGenerator(nn.Module):
    """
    Standard pix2pix-style U-Net, 8 down / 8 up blocks for 256x256 input.
    in_channels=1 (L channel), out_channels=2 (ab channels)
    """
    def __init__(self, in_channels=1, out_channels=2, features=64):
        super().__init__()

        # ---- Encoder (downsampling) ----
        self.down1 = UNetDown(in_channels, features, use_norm=False)      # 256 -> 128
        self.down2 = UNetDown(features, features * 2)                     # 128 -> 64
        self.down3 = UNetDown(features * 2, features * 4)                 # 64  -> 32
        self.down4 = UNetDown(features * 4, features * 8)                 # 32  -> 16
        self.down5 = UNetDown(features * 8, features * 8)                 # 16  -> 8
        self.down6 = UNetDown(features * 8, features * 8)                 # 8   -> 4
        self.down7 = UNetDown(features * 8, features * 8)                 # 4   -> 2
        self.bottleneck = UNetDown(features * 8, features * 8, use_norm=False)  # 2 -> 1

        # ---- Decoder (upsampling), with dropout in the first few layers
        # (standard pix2pix trick to reduce overfitting / add stochasticity) ----
        self.up1 = UNetUp(features * 8, features * 8, use_dropout=True)   # 1  -> 2
        self.up2 = UNetUp(features * 16, features * 8, use_dropout=True)  # 2  -> 4
        self.up3 = UNetUp(features * 16, features * 8, use_dropout=True)  # 4  -> 8
        self.up4 = UNetUp(features * 16, features * 8)                    # 8  -> 16
        self.up5 = UNetUp(features * 16, features * 4)                    # 16 -> 32
        self.up6 = UNetUp(features * 8, features * 2)                     # 32 -> 64
        self.up7 = UNetUp(features * 4, features)                         # 64 -> 128

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(features * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()   # output constrained to [-1, 1], matching our ab normalization
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)

        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


if __name__ == "__main__":
    model = UNetGenerator()
    dummy_input = torch.randn(2, 1, 256, 256)
    out = model(dummy_input)
    print("Output shape:", out.shape)   # expect (2, 2, 256, 256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
