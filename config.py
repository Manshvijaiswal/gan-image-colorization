"""
config.py
---------
All hyperparameters and paths in one place so you never have to hunt
through train.py to change a setting. Import this everywhere.
"""

import torch

class Config:
    # ---- Paths ----
    TRAIN_DIR = "data/raw/train"
    VAL_DIR = "data/raw/val"
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "outputs"

    # ---- Device ----
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Image ----
    IMAGE_SIZE = 256          # resize all images to 256x256

    # ---- Training ----
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LR_G = 2e-4                # generator learning rate
    LR_D = 2e-4                # discriminator learning rate
    BETA1 = 0.5                 # Adam optimizer momentum term (standard for GANs)
    BETA2 = 0.999

    # ---- Loss weights ----
    LAMBDA_L1 = 100.0           # weight for L1 (pixel) loss -- dominant term
    LAMBDA_ADV = 1.0            # weight for adversarial loss
    LAMBDA_PERCEPTUAL = 10.0    # weight for perceptual (VGG) loss
    USE_PERCEPTUAL_LOSS = True  # enabled: addresses speckled/patchy color noise on flat regions

    # ---- Model ----
    GEN_FEATURES = 64           # base number of filters in generator
    DISC_FEATURES = 64          # base number of filters in discriminator

    # ---- Misc ----
    SEED = 42
    SAVE_EVERY = 5              # save checkpoint every N epochs
    LOG_EVERY = 100             # print loss every N batches
    NUM_WORKERS = 2             # dataloader workers (Colab free tier: keep at 2)

cfg = Config()
