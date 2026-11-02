# ==============================================================================
# DOCKERFILE SOUVERAIN ASIN — CHAÎNE IA VISION, FFMPEG NVDEC & OPTIMISATION GPU
# Auteur : Ing. Christophe WAVOEKE (Candidature Ingénieur IA Vision - ASIN Bénin)
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="Ing. Christophe WAVOEKE <christophewavoeke18@gmail.com>"
LABEL organization="ASIN Bénin - Pôle Systèmes d'Information et Digitalisation (PSID)"
LABEL description="Chaîne d'analyse vidéo VLM open-weights, décodage GPU FFmpeg et supervision vLLM"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# Installation de Python 3.11, FFmpeg avec accélération matérielle et librairies système OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Utilisateur non-root pour sécurité souveraine (Conformité PSSI ASIN)
RUN useradd -m -u 10001 asin_vision_user && chown -R asin_vision_user:asin_vision_user /app
USER asin_vision_user

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

