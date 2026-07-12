# Deployment Guide

This gets you two things for your resume/portfolio: a clean **public GitHub
repo**, and a **live, working demo URL** anyone can click.

The one tricky part: your trained model (`generator_final.pth`) is likely
50-200MB+, and GitHub blocks files over 100MB in a normal push. We handle
this by hosting the weights separately (GitHub Releases) and downloading
them at app startup — you never `git push` the weight file itself.

---

## Part 1 — Push the code to GitHub (without the huge checkpoint file)

1. Create a new repo on GitHub (e.g. `gan-image-colorization`), don't
   initialize it with a README (you already have one).

2. From your project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: GAN image colorization"
   git branch -M main
   git remote add origin https://github.com/<your-username>/gan-image-colorization.git
   git push -u origin main
   ```
   The `.gitignore` already excludes `checkpoints/*.pth`, so this push will
   be small and fast — it contains code only, not model weights.

3. **Upload your trained weights as a GitHub Release asset** (this is the
   standard way to distribute large binaries alongside a repo):
   - On your repo's GitHub page: **Releases → Create a new release**
   - Tag it `v1.0`, title "Trained model weights"
   - Drag `generator_final.pth` into the assets upload box
   - Publish the release
   - Right-click the uploaded file link → copy the direct download URL
     (looks like `https://github.com/<user>/<repo>/releases/download/v1.0/generator_final.pth`)
   - Save that URL — you'll need it in Part 2.

---

## Part 2 — Deploy the live demo

You have two apps in this repo (`web/server.py` = the custom Flask website,
`app.py` = the Streamlit alternative). Pick ONE to deploy publicly — the
custom website is the stronger portfolio piece; Streamlit is faster to set up.

### Option A (recommended): Hugging Face Spaces, Docker SDK — deploys the custom website

1. Create a free account at huggingface.co, then **New Space**.
2. Choose **Docker** as the Space SDK — it will build directly from the `Dockerfile` already in the repo, which serves the custom website by default.
3. In your Space's **Settings → Variables and secrets**, add:
   - `GENERATOR_WEIGHTS_URL` = the GitHub Release URL from Part 1
4. Either connect the Space directly to your GitHub repo (Settings → "Sync with a GitHub repo"), or push your code to the Space's own git remote (Hugging Face gives you a `git remote add` command on the Space page).
5. The Space builds automatically. On first boot, `ensure_weights_available()` in `web/server.py` downloads your weights from the Release URL — no large file in the repo or the Space itself.
6. Your live URL will be `https://huggingface.co/spaces/<your-username>/<space-name>` — put this directly on your resume/GitHub README.

### Option B: Streamlit Community Cloud — deploys app.py, simplest setup

1. Push your code to GitHub (Part 1).
2. Go to share.streamlit.io, sign in with GitHub, click **New app**, select your repo and `app.py` as the entrypoint.
3. Since Streamlit Cloud doesn't have a GitHub-Release-downloading step built in like `web/server.py` does, add this snippet to the top of `app.py` (after the imports) so the checkpoint downloads automatically on first run:
   ```python
   import os, urllib.request
   CKPT = "checkpoints/generator_final.pth"
   CKPT_URL = "PASTE_YOUR_GITHUB_RELEASE_URL_HERE"
   if not os.path.exists(CKPT):
       os.makedirs("checkpoints", exist_ok=True)
       urllib.request.urlretrieve(CKPT_URL, CKPT)
   ```
4. Deploy. You'll get a URL like `https://<your-app>.streamlit.app`.

### Option C: Render / Railway (Docker) — either app, more control

1. Push to GitHub (Part 1).
2. On Render.com: **New → Web Service**, connect your repo, choose "Docker" as the environment (uses the included `Dockerfile`).
3. Add an environment variable `GENERATOR_WEIGHTS_URL` with your Release URL (if deploying `web/server.py`) as in Option A.
4. Deploy — Render gives you a `https://<app>.onrender.com` URL.
   Note: Render's free tier spins down after inactivity, so the first
   request after idle time will be slow (~30-60s cold start) — fine for a
   portfolio link, just mention it if a recruiter tries it cold.

---

## Part 3 — Polish for resume/portfolio presentation

- Add 1-2 sentences + the live link to your resume/LinkedIn under Projects.
- On the GitHub repo itself: make sure the README's `sample_results.png` and live-demo link are filled in (not placeholders).
- Consider adding a short screen-recording GIF of the site in action to the top of the README — this is often what actually gets a recruiter's attention in 5 seconds of scrolling.
- Pin the repo on your GitHub profile (Profile → Customize your pins).

---

## Troubleshooting

- **"Model checkpoint not found" on the deployed app:** double-check the `GENERATOR_WEIGHTS_URL` env var is set correctly and is a *direct* download link (test it by pasting into a private/incognito browser tab — it should immediately download a `.pth` file, not show a GitHub webpage).
- **Deployed app is very slow:** most free hosting tiers are CPU-only. A single-image inference on CPU for a 256×256 U-Net typically takes a few seconds — normal. If it's taking 30+ seconds, check you're not accidentally reloading the model from disk on every request (both `web/server.py` and `app.py` are written to load the model once at startup / via `st.cache_resource`).
- **Out of memory during Docker build:** reduce `requirements.txt` to only what the *deployed app* needs (Streamlit/Flask + torch + inference deps) — training-only packages like `pytorch-fid` aren't needed in the deployed image and can be split into a separate `requirements-dev.txt`.
