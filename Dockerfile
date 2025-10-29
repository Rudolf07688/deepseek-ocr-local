# DeepSeek-OCR Docker Image
# Using uv for fast package management
# Optimized for CUDA 13.0 / RTX 3080 Ti (12GB VRAM)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including Python and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral's fast Python package installer)
RUN pip3 install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Install dependencies
# Use PyTorch's CUDA 12.1 wheel index for torch packages
RUN uv pip install --system \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies from standard PyPI
RUN uv pip install --system \
    transformers==4.45.0 \
    tokenizers \
    einops \
    addict \
    easydict \
    pillow \
    pdf2image \
    tqdm \
    matplotlib

# Copy application code
COPY deepseek_ocr_pdf.py .
COPY advanced_examples.py .
COPY quick_test.py .

# Create directories
RUN mkdir -p /app/input /app/output /app/models

# Set model cache location
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Expose CUDA devices
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Use uv to run Python with proper environment
CMD ["uv", "run", "python3", "-c", "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'PyTorch: {torch.__version__}')"]
