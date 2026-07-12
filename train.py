"""
train.py
---------
Main training loop for the GAN-based colorization model.

TRAINING LOGIC (read this before running):
Each batch, we do TWO separate backward passes:
  1. Update Discriminator: show it a real (L, real_ab) pair and a fake
     (L, fake_ab.detach()) pair. It should learn to tell them apart.
     We .detach() the fake ab so gradients don't flow into G here.
  2. Update Generator: it wants D to be FOOLED by (L, fake_ab), AND its
     output should be close to the real ab (L1 loss), AND (optionally)
     look perceptually similar (VGG loss).

We use LSGAN loss (MSE instead of BCE) because it's more stable and
suffers less from vanishing gradients than the original vanilla GAN loss.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import ColorizationDataset
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from models.perceptual import VGGPerceptualLoss
from utils.color_utils import lab_batch_to_rgb_torch
from utils.visualize import save_sample_grid


def weights_init(m):
    """Standard DCGAN-style weight initialization -- helps GAN training stability."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(resume_epoch=0):
    torch.manual_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ---- Data ----
    train_ds = ColorizationDataset(cfg.TRAIN_DIR, image_size=cfg.IMAGE_SIZE, split="train")
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)

    val_ds = ColorizationDataset(cfg.VAL_DIR, image_size=cfg.IMAGE_SIZE, split="val")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=2)

    # ---- Models ----
    G = UNetGenerator(in_channels=1, out_channels=2, features=cfg.GEN_FEATURES).to(cfg.DEVICE)
    D = PatchDiscriminator(in_channels=3, features=cfg.DISC_FEATURES).to(cfg.DEVICE)

    # ---- Optimizers ----
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))

    if resume_epoch > 0:
        # ---- Resume from a saved checkpoint instead of random init ----
        gen_path = os.path.join(cfg.CHECKPOINT_DIR, f"generator_epoch{resume_epoch}.pth")
        disc_path = os.path.join(cfg.CHECKPOINT_DIR, f"discriminator_epoch{resume_epoch}.pth")
        print(f"Resuming from epoch {resume_epoch}")
        print(f"  Loading generator:     {gen_path}")
        print(f"  Loading discriminator: {disc_path}")
        G.load_state_dict(torch.load(gen_path, map_location=cfg.DEVICE))
        D.load_state_dict(torch.load(disc_path, map_location=cfg.DEVICE))

        # Optimizer state (Adam momentum) is only available if we saved it
        # previously (see the checkpoint block below). Older checkpoints that
        # only stored model weights will simply start Adam's momentum fresh --
        # this has minimal impact this late in training.
        opt_g_path = os.path.join(cfg.CHECKPOINT_DIR, f"opt_generator_epoch{resume_epoch}.pth")
        opt_d_path = os.path.join(cfg.CHECKPOINT_DIR, f"opt_discriminator_epoch{resume_epoch}.pth")
        if os.path.exists(opt_g_path) and os.path.exists(opt_d_path):
            opt_G.load_state_dict(torch.load(opt_g_path, map_location=cfg.DEVICE))
            opt_D.load_state_dict(torch.load(opt_d_path, map_location=cfg.DEVICE))
            print("  Optimizer states restored.")
        else:
            print("  No saved optimizer state found -- Adam momentum will reinitialize.")
    else:
        # ---- Fresh run: standard DCGAN-style weight initialization ----
        G.apply(weights_init)
        D.apply(weights_init)

    perceptual_loss_fn = None
    if cfg.USE_PERCEPTUAL_LOSS:
        perceptual_loss_fn = VGGPerceptualLoss().to(cfg.DEVICE)

    # ---- Losses ----
    adversarial_loss = nn.MSELoss()   # LSGAN-style loss (more stable than BCE)
    l1_loss = nn.L1Loss()

    print(f"Training on device: {cfg.DEVICE}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    start_epoch = resume_epoch + 1
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        epoch_start = time.time()
        G.train()
        D.train()

        for batch_idx, (L, real_ab) in enumerate(train_loader):
            L, real_ab = L.to(cfg.DEVICE), real_ab.to(cfg.DEVICE)
            b = L.size(0)

            # =========================================================
            # 1) Train Discriminator
            # =========================================================
            opt_D.zero_grad()

            fake_ab = G(L)

            pred_real = D(L, real_ab)
            pred_fake = D(L, fake_ab.detach())   # detach: don't backprop into G here

            real_labels = torch.ones_like(pred_real, device=cfg.DEVICE)
            fake_labels = torch.zeros_like(pred_fake, device=cfg.DEVICE)

            loss_d_real = adversarial_loss(pred_real, real_labels)
            loss_d_fake = adversarial_loss(pred_fake, fake_labels)
            loss_D = 0.5 * (loss_d_real + loss_d_fake)

            loss_D.backward()
            opt_D.step()

            # =========================================================
            # 2) Train Generator
            # =========================================================
            opt_G.zero_grad()

            pred_fake_for_g = D(L, fake_ab)
            loss_g_adv = adversarial_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = l1_loss(fake_ab, real_ab) * cfg.LAMBDA_L1

            loss_G = cfg.LAMBDA_ADV * loss_g_adv + loss_g_l1
            loss_perc_val = 0.0

            if cfg.USE_PERCEPTUAL_LOSS:
                fake_rgb = lab_batch_to_rgb_torch(L, fake_ab)
                real_rgb = lab_batch_to_rgb_torch(L, real_ab)
                loss_perc = perceptual_loss_fn(fake_rgb, real_rgb) * cfg.LAMBDA_PERCEPTUAL
                loss_G = loss_G + loss_perc
                loss_perc_val = loss_perc.item()

            loss_G.backward()
            opt_G.step()

            if batch_idx % cfg.LOG_EVERY == 0:
                print(f"[Epoch {epoch}/{cfg.NUM_EPOCHS}] "
                      f"[Batch {batch_idx}/{len(train_loader)}] "
                      f"D_loss: {loss_D.item():.4f} | "
                      f"G_loss: {loss_G.item():.4f} "
                      f"(adv: {loss_g_adv.item():.4f}, l1: {loss_g_l1.item():.4f}, "
                      f"perc: {loss_perc_val:.4f})")

        print(f"Epoch {epoch} finished in {time.time() - epoch_start:.1f}s")

        # ---- Save sample colorizations for visual sanity check ----
        G.eval()
        with torch.no_grad():
            val_L, val_ab = next(iter(val_loader))
            val_L = val_L.to(cfg.DEVICE)
            fake_val_ab = G(val_L)
            save_sample_grid(val_L.cpu(), val_ab.cpu(), fake_val_ab.cpu(),
                              save_path=os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch:03d}.png"))

        # ---- Checkpoint ----
        if epoch % cfg.SAVE_EVERY == 0:
            torch.save(G.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"generator_epoch{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pth"))
            torch.save(opt_G.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"opt_generator_epoch{epoch}.pth"))
            torch.save(opt_D.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"opt_discriminator_epoch{epoch}.pth"))
            print(f"Saved checkpoint at epoch {epoch}")

    # ---- Final save ----
    torch.save(G.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "discriminator_final.pth"))
    print("Training complete. Final models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=0,
                         help="Epoch number to resume from (must match a saved checkpoint). "
                              "0 (default) starts a fresh training run.")
    args = parser.parse_args()
    train(resume_epoch=args.resume)
