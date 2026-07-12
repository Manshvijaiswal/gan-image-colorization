"""
app.py
-------
Streamlit web app for the GAN colorization demo -- portfolio/resume-ready
version with a proper landing layout, architecture summary, and example
gallery, in addition to the core upload -> colorize workflow.

Run locally:
    streamlit run app.py

Deploy (pick one):
  - Hugging Face Spaces (recommended -- free, permanent public URL, native
    Streamlit support, good for resumes/portfolios)
  - Streamlit Community Cloud (free, connects directly to your GitHub repo)
  - Render / Railway / Fly.io (containerized deployment via Dockerfile)

See DEPLOYMENT.md for step-by-step instructions for each.
"""

import io
import os
import glob
import streamlit as st
from PIL import Image

from inference import load_generator, colorize_image
from config import cfg

st.set_page_config(
    page_title="GAN Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal custom styling -- keeps the default Streamlit look but tightens
# spacing and adds a bit of visual polish (card-like image containers,
# consistent header sizing) without needing a full custom theme.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stImage img { border-radius: 8px; border: 1px solid #e6e6e6; }
    h1 { font-weight: 700; }
    .subtitle { color: #6b7280; font-size: 1.05rem; margin-top: -0.6rem; }
    .footer { color: #9ca3af; font-size: 0.85rem; text-align: center; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar -- project info, architecture summary, model settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        "A **conditional GAN** that colorizes grayscale photographs, "
        "built with PyTorch."
    )
    st.markdown("**Architecture**")
    st.markdown(
        "- U-Net Generator (skip connections)\n"
        "- PatchGAN Discriminator (70×70 patches)\n"
        "- LAB color space (predicts a,b from L)\n"
        "- Adversarial (LSGAN) + L1 + VGG Perceptual loss"
    )
    st.markdown("[View source on GitHub](https://github.com/YOUR_USERNAME/gan-image-colorization)")

    st.divider()
    st.subheader("Model settings")
    checkpoint_path = st.text_input(
        "Checkpoint path", value="checkpoints/generator_final.pth",
        help="Path to the trained generator weights (.pth file)."
    )


@st.cache_resource(show_spinner=False)
def get_model(checkpoint_path):
    return load_generator(checkpoint_path)


def run_colorization(input_image, checkpoint_path):
    """Shared colorization logic used by both the uploader and example gallery."""
    G = get_model(checkpoint_path)
    buf = io.BytesIO()
    input_image.save(buf, format="PNG")
    buf.seek(0)
    temp_path = "temp_upload.png"
    with open(temp_path, "wb") as f:
        f.write(buf.read())
    return colorize_image(G, temp_path, image_size=cfg.IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎨 GAN-Based Image Colorization")
st.markdown(
    '<p class="subtitle">Upload a black-and-white photo and a U-Net + '
    "PatchGAN conditional GAN predicts a realistic, plausible coloring.</p>",
    unsafe_allow_html=True,
)
st.write("")

tab_demo, tab_examples, tab_about = st.tabs(["🖼️ Try it", "✨ Example results", "📖 How it works"])

# ---------------------------------------------------------------------------
# Tab 1: Main upload -> colorize demo
# ---------------------------------------------------------------------------
with tab_demo:
    uploaded_file = st.file_uploader(
        "Upload an image (grayscale or color -- color images will have their color discarded first)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(input_image, use_container_width=True)

        if st.button("🎨 Colorize", type="primary"):
            with st.spinner("Running generator..."):
                try:
                    result = run_colorization(input_image, checkpoint_path)
                    with col2:
                        st.subheader("Colorized Output")
                        st.image(result, use_container_width=True)

                    out_buf = io.BytesIO()
                    result.save(out_buf, format="PNG")
                    st.download_button("⬇ Download result", out_buf.getvalue(),
                                        file_name="colorized.png", mime="image/png")
                except FileNotFoundError:
                    st.error(
                        f"Checkpoint not found at `{checkpoint_path}`. "
                        "Train the model first (see `train.py`), or update the path "
                        "in the sidebar -- for a deployed demo, make sure the checkpoint "
                        "file is included in the deployment (see DEPLOYMENT.md)."
                    )
    else:
        st.info("👆 Upload an image to get started, or check the **Example results** tab.")

# ---------------------------------------------------------------------------
# Tab 2: Pre-computed example gallery
# ---------------------------------------------------------------------------
with tab_examples:
    st.write(
        "Sample outputs from the trained model on held-out test images "
        "(grayscale input vs. generated colorization)."
    )
    example_dir = "assets/examples"
    example_imgs = sorted(glob.glob(os.path.join(example_dir, "*.png"))) + \
                   sorted(glob.glob(os.path.join(example_dir, "*.jpg")))

    if example_imgs:
        cols = st.columns(2)
        for i, img_path in enumerate(example_imgs):
            with cols[i % 2]:
                st.image(img_path, use_container_width=True)
    else:
        st.warning(
            f"No example images found in `{example_dir}/`. "
            "Add a few sample comparison grids there (e.g. exported from "
            "`outputs/epoch_XXX.png` during training) so visitors can see "
            "results without uploading their own image."
        )

# ---------------------------------------------------------------------------
# Tab 3: Explanation for non-technical visitors (recruiters, etc.)
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### The problem
A black-and-white photo only records brightness. Adding color back is
inherently ambiguous -- a gray shirt could plausibly be red, blue, or
green -- so this isn't a simple lookup, it's a *learned, generative* task.

### The approach
This project uses a **conditional Generative Adversarial Network (cGAN)**,
following the architecture introduced in *pix2pix* (Isola et al., 2017):

- **Generator (U-Net):** takes the grayscale *L* (lightness) channel and
  predicts the *a, b* (color) channels in LAB color space. Skip connections
  preserve fine edges so color aligns precisely with object boundaries.
- **Discriminator (PatchGAN):** instead of judging the whole image as
  real/fake in one shot, it classifies overlapping 70×70 patches, giving a
  much richer training signal and sharper, more locally consistent color.
- **Loss functions:** an adversarial loss pushes for realistic, vivid color
  (avoiding the washed-out results typical of pixel-loss-only models); an
  L1 loss anchors predictions to ground truth; a VGG **perceptual loss**
  keeps textures (skin, fabric, foliage) smooth and semantically coherent.

### Why not just use a plain CNN?
A CNN trained only to minimize pixel error tends to predict the *average*
of all plausible colors for an ambiguous region -- which is a dull,
desaturated compromise. The adversarial component here specifically
rewards *realistic-looking* color, not just *numerically close* color.
""")

st.markdown(
    '<div class="footer">Built with PyTorch + Streamlit · '
    '<a href="https://github.com/YOUR_USERNAME/gan-image-colorization">Source on GitHub</a></div>',
    unsafe_allow_html=True,
)
