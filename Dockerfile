# Use NVIDIA PyTorch runtime compatible with CUDA 12.x
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="7.5"

# Create cache directory and set permissions
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Create a volume for persistent model cache
VOLUME ["/app/model_cache"]

# Pre-download and cache the SeamlessM4T model (only if cache is empty)
RUN python -c "import os; \
    try: \
        from transformers import SeamlessM4TModel, AutoProcessor; \
        print('Downloading SeamlessM4T model and processor...'); \
        model = SeamlessM4TModel.from_pretrained('facebook/seamless-m4t-large'); \
        processor = AutoProcessor.from_pretrained('facebook/seamless-m4t-large'); \
        print('Model download completed successfully'); \
    except Exception as e: \
        print(f'Model download failed: {e}'); \
        print('Model will be downloaded at runtime instead'); \
        exit(0)"

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["python", "backend/main.py"]