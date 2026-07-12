"""
web/server.py
--------------
Flask backend for the deployed web demo.

Serves:
  GET  /               -> the custom frontend (index.html)
  POST /api/colorize   -> accepts an uploaded image, returns a colorized PNG

Model weights are loaded once at startup (not per-request) for speed.
If GENERATOR_WEIGHTS_URL is set, the weight file is downloaded on first
boot (used for deployment, since trained weights are too large for a
normal git push -- see README for hosting instructions).
"""

import os
import io
import sys
import base64
import urllib.request

from flask import Flask, request, jsonify, send_from_directory
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generator import UNetGenerator
from inference import colorize_image_from_pil

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
WEIGHTS_PATH = os.environ.get("GENERATOR_WEIGHTS_PATH", "checkpoints/generator_final.pth")
WEIGHTS_URL = os.environ.get("GENERATOR_WEIGHTS_URL", "")   # e.g. a GitHub Release asset URL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")


def ensure_weights_available():
    if os.path.exists(WEIGHTS_PATH):
        return
    if not WEIGHTS_URL:
        raise FileNotFoundError(
            f"No weights found at {WEIGHTS_PATH} and GENERATOR_WEIGHTS_URL is not set. "
            "Either place generator_final.pth at that path, or set the env var to a "
            "direct download URL (see README -> Deployment)."
        )
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    print(f"Downloading generator weights from {WEIGHTS_URL} ...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    print("Weights downloaded.")


print("Loading generator model...")
ensure_weights_available()
generator = UNetGenerator(in_channels=1, out_channels=2, features=64).to(DEVICE)
generator.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
generator.eval()
print(f"Model loaded on {DEVICE}.")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/colorize", methods=["POST"])
def colorize():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under field name 'image'."}), 400

    file = request.files["image"]
    try:
        from PIL import Image
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    try:
        result_img = colorize_image_from_pil(generator, img, device=DEVICE)
    except Exception as e:
        return jsonify({"error": f"Colorization failed: {e}"}), 500

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({"colorized_png_base64": encoded})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
