# 🎨 GAN-Based Image Colorization

Automatic colorization of black-and-white photographs using a conditional GAN — a U-Net Generator and PatchGAN Discriminator operating in LAB color space, trained adversarially with L1 and VGG perceptual losses.

**🔗 Live demo:** [add your deployed URL here after following DEPLOYMENT.md]

![Sample Results](assets/sample_results.png)
*Left to right: grayscale input → ground truth → model output (epoch 70, trained on COCO val2017)*

## Highlights

- **Conditional GAN architecture** (pix2pix-style): U-Net generator with skip connections + PatchGAN discriminator conditioned on the input luminance channel
- **LAB color space** decomposition — the network predicts only the 2 chrominance channels (a, b), keeping the luminance (L) channel untouched
- **Hybrid loss function**: LSGAN adversarial loss + weighted L1 pixel loss + VGG-16 perceptual loss for texture/semantic consistency
- **Fully custom, differentiable LAB↔RGB conversion in PyTorch** (GPU-resident, batched) — avoids the common CPU-bottleneck mistake of looping per-image through `skimage` inside the training loop
- Fixed a **checkerboard-artifact bug** from `ConvTranspose2d` decoder layers by switching to Upsample+Conv, a known GAN stabilization technique
- Resumable training pipeline (model + optimizer state checkpointing) — trained across multiple sessions on Google Colab
- **Two deployment options included:** a custom-designed Flask + HTML/CSS/JS web app (`web/`), and a Streamlit demo (`app.py`) for quick iteration

## Architecture

```
Grayscale L channel ──► U-Net Generator (encoder-decoder, skip connections) ──► predicted a,b channels
                                                                                        │
                              concat with L → fake LAB image                          │
                                       │                                               │
                                       ▼                                               │
                          PatchGAN Discriminator ◄── real LAB image ────────────────────┘
                       (30x30 grid of real/fake patch predictions)
```

## Tech Stack
`Python` · `PyTorch` · `torchvision` (VGG16 perceptual features) · `scikit-image` (LAB conversion for I/O) · `Streamlit` (deployment) · trained on Google Colab (T4 GPU)

## Results

Trained for 70 epochs on a 4,500-image subset of COCO val2017 (90/10 train/val split), 256×256 resolution.

| Stage | Observation |
|---|---|
| Epochs 1-10 | Model converges to a "safe" muted/olive color palette (expected pixel-loss behavior) |
| Epoch 11 | Diagnosed and fixed a checkerboard artifact caused by transposed convolutions |
| Epoch 50 | Adversarial training produces vivid, mostly-accurate color, but speckled noise appears on flat/low-texture regions (skin, walls) |
| Epoch 70 (final) | Enabled VGG perceptual loss — speckled noise resolved, textures smooth and coherent; some hue inaccuracy remains on visually ambiguous objects (clothing, man-made objects with less consistent training-data color priors) |

**Known limitation:** the model reliably colors high-frequency, common categories (foliage, animal fur, skin tones, sky) but occasionally misassigns hue on objects with high real-world color variance (e.g., a magenta umbrella predicted as green) — a documented trade-off in colorization literature between perceptual/textural realism and exact hue reproduction.

## Project Structure
```
gan_colorization/
├── data/dataset.py           # RGB -> LAB dataset pipeline
├── models/
│   ├── generator.py          # U-Net Generator
│   ├── discriminator.py      # PatchGAN Discriminator
│   └── perceptual.py         # VGG16 perceptual loss
├── utils/
│   ├── color_utils.py        # RGB <-> LAB conversion (incl. differentiable GPU version)
│   ├── visualize.py
│   └── metrics.py            # PSNR / SSIM / FID
├── web/
│   ├── server.py              # Flask backend (POST /api/colorize)
│   └── static/index.html      # custom-designed frontend
├── train.py                  # training loop (with resume support)
├── evaluate.py                # PSNR/SSIM/FID evaluation
├── inference.py               # single-image colorization
├── app.py                     # Streamlit demo (alternative quick-deploy option)
├── Dockerfile                 # containerized deployment
└── config.py                  # hyperparameters
```

## Setup & Usage

```bash
git clone https://github.com/<your-username>/gan-image-colorization.git
cd gan-image-colorization
pip install -r requirements.txt

# Train
python train.py

# Resume training from a checkpoint
python train.py --resume 50

# Colorize an image from the command line
python inference.py --checkpoint checkpoints/generator_final.pth --input photo.jpg --output result.jpg

# Run the custom website locally (Flask backend + static frontend)
python web/server.py
# then open http://localhost:7860

# OR run the Streamlit demo instead
streamlit run app.py
```

**Deploying publicly (free, permanent URL for your resume):** see [`DEPLOYMENT.md`](DEPLOYMENT.md) for step-by-step instructions covering Hugging Face Spaces, Render, and GitHub setup (including how to handle the large model checkpoint file, which can't go through a normal `git push`).

## References
- Isola et al., 2017 — *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix)
- Zhang, Isola & Efros, 2016 — *Colorful Image Colorization*
- Goodfellow et al., 2014 — *Generative Adversarial Networks*
