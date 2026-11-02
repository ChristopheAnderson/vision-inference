FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="Ing. Christophe WAVOEKE <christophewavoeke18@gmail.com>"
LABEL description="VisionInference Hub — VLM Video Analytics & GPU Optimization (FFmpeg NVDEC + PyTorch + vLLM AWQ)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=42

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
