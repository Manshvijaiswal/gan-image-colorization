# Dockerfile for the GAN Image Colorization Streamlit app.
# Works on Render, Railway, Fly.io, Hugging Face Spaces (Docker SDK), etc.

FROM python:3.10-slim

WORKDIR /app

# System deps needed by opencv/pillow/scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/api/health || exit 1

# Serves the custom Flask + HTML website (web/server.py) by default.
# To deploy the Streamlit alternative instead, replace this line with:
#   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
ENTRYPOINT ["python", "web/server.py"]
