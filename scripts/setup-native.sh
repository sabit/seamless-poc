#!/bin/bash

# Native VM setup script for SeamlessStreaming Translation Service
# Run this script on your GCP VM to set up the service without Docker

set -e

echo "🚀 Setting up SeamlessStreaming Translation Service (Native VM)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on the right system
print_status "Checking system requirements..."

# Check CUDA
if nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU is available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_error "NVIDIA GPU not detected. Please ensure CUDA drivers are installed."
    exit 1
fi

# Check Python 3
if python3 --version &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python 3 is available: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    python3-pyaudio \
    curl \
    git

print_success "System dependencies installed"

# Create Python virtual environment
VENV_DIR="$HOME/seamless-venv"
print_status "Creating Python virtual environment at $VENV_DIR..."

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists, removing old one..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

print_success "Virtual environment created and activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA
print_status "Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install only missing packages (respecting gold image versions)
print_status "Installing missing application dependencies..."
print_status "Using gold image versions of: fastapi, uvicorn, websockets, aiofiles, torch, numpy, scipy"

pip install \
    python-multipart \
    torchaudio \
    transformers>=4.37.0 \
    accelerate \
    librosa \
    soundfile \
    sentencepiece \
    huggingface-hub

print_success "Python packages installed"

# Create model cache directory
MODEL_CACHE_DIR="$HOME/seamless-model-cache"
print_status "Creating model cache directory at $MODEL_CACHE_DIR..."
mkdir -p "$MODEL_CACHE_DIR"

# Set environment variables
print_status "Setting up environment variables..."
cat > "$HOME/seamless-env.sh" << EOF
#!/bin/bash
# SeamlessStreaming environment setup

export PYTHONPATH="$PWD"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_HOME="$MODEL_CACHE_DIR"
export CUDA_VISIBLE_DEVICES=0

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "🔧 SeamlessStreaming environment activated"
echo "📂 Working directory: \$PWD"
echo "🤖 Model cache: \$MODEL_CACHE_DIR"
echo "🐍 Python: \$(which python3)"
echo "⚡ CUDA available: \$(python3 -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x "$HOME/seamless-env.sh"

print_success "Environment setup complete!"

# Pre-download the model (optional)
print_status "Do you want to download the SeamlessM4T model now? (y/n)"
read -r DOWNLOAD_MODEL

if [[ $DOWNLOAD_MODEL =~ ^[Yy]$ ]]; then
    print_status "Downloading SeamlessM4T model... (this may take several minutes)"
    source "$HOME/seamless-env.sh"
    
    python3 -c "
from transformers import SeamlessM4TModel, AutoProcessor
print('Downloading model...')
model = SeamlessM4TModel.from_pretrained('facebook/seamless-m4t-large')
processor = AutoProcessor.from_pretrained('facebook/seamless-m4t-large')
print('Model download completed!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Model downloaded and cached"
    else
        print_warning "Model download failed, but will be downloaded on first run"
    fi
fi

print_success "🎉 Native VM setup complete!"
echo ""
echo "🚀 To run the service:"
echo "   1. Activate environment: source ~/seamless-env.sh"
echo "   2. Start service: python3 backend/main.py"
echo ""
echo "📱 The service will be available at: http://localhost:7860"
echo ""
echo "🔧 Useful commands:"
echo "   View environment: source ~/seamless-env.sh"
echo "   Test GPU: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "   Check model cache: ls -la $MODEL_CACHE_DIR"