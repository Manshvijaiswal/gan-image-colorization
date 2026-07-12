FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/api/health || exit 1

ENTRYPOINT ["python", "web/server.py"]
