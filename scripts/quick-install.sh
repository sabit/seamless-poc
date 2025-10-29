#!/bin/bash

# Quick install script for missing packages only
# Uses gold image pre-installed versions and adds only what's needed

set -e

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

echo "🚀 Quick Install for SeamlessStreaming (Gold Image Compatible)"
echo "============================================================="

# Check Python
if ! python3 --version &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
print_status "Using: $PYTHON_VERSION"

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    print_error "pip not found"
    exit 1
fi

# Show what's already installed in gold image
print_status "Gold image already has:"
echo "  ✅ fastapi (v0.115.12)"
echo "  ✅ uvicorn (v0.34.0)" 
echo "  ✅ websockets (v15.0.1)"
echo "  ✅ aiofiles (v22.1.0)"
echo "  ✅ torch (v2.4.0+cu124)"
echo "  ✅ numpy (v1.25.2)"
echo "  ✅ scipy (v1.11.4)"
echo ""

# Create model cache directory
MODEL_CACHE_DIR="$HOME/seamless-model-cache"
print_status "Creating model cache directory: $MODEL_CACHE_DIR"
mkdir -p "$MODEL_CACHE_DIR"

# Install only missing packages
print_status "Installing missing packages..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_success "Using virtual environment: $VIRTUAL_ENV"
else
    print_warning "Not in a virtual environment - installing to user directory"
    PIP_FLAGS="--user"
fi

# Install missing packages one by one for better error handling
PACKAGES=(
    "python-multipart"
    "torchaudio" 
    "transformers>=4.37.0"
    "accelerate"
    "librosa"
    "soundfile" 
    "sentencepiece"
    "huggingface-hub"
)

for package in "${PACKAGES[@]}"; do
    print_status "Installing $package..."
    if pip install $PIP_FLAGS "$package"; then
        print_success "$package installed"
    else
        print_error "Failed to install $package"
        exit 1
    fi
done

print_success "All missing packages installed!"

# Set up environment variables
print_status "Setting up environment..."

cat > "$HOME/seamless-env-quick.sh" << EOF
#!/bin/bash
# Quick SeamlessStreaming environment setup

export PYTHONPATH="\$PWD"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_HOME="$MODEL_CACHE_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "🔧 SeamlessStreaming environment activated (Gold Image Compatible)"
echo "📂 Working directory: \$PWD"
echo "🤖 Model cache: $MODEL_CACHE_DIR"
echo "🐍 Python: \$(which python3)"
echo "⚡ CUDA available: \$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'checking...')"
EOF

chmod +x "$HOME/seamless-env-quick.sh"

# Test installation
print_status "Testing installation..."

if python3 -c "
import sys
try:
    # Test gold image packages
    import fastapi, uvicorn, websockets, aiofiles, torch, numpy, scipy
    print('✅ Gold image packages working')
    
    # Test newly installed packages  
    import transformers, librosa, soundfile, sentencepiece
    print('✅ New packages installed correctly')
    
    # Test CUDA
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️ CUDA not available, will use CPU')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"; then
    print_success "Installation test passed!"
else
    print_error "Installation test failed"
    exit 1
fi

# Pre-download model option
print_status "Do you want to download the SeamlessM4T model now? (y/n)"
read -r DOWNLOAD_MODEL

if [[ $DOWNLOAD_MODEL =~ ^[Yy]$ ]]; then
    print_status "Downloading SeamlessM4T model..."
    source "$HOME/seamless-env-quick.sh"
    
    python3 -c "
from transformers import SeamlessM4TModel, AutoProcessor
print('Downloading SeamlessM4T model...')
model = SeamlessM4TModel.from_pretrained('facebook/seamless-m4t-large')
processor = AutoProcessor.from_pretrained('facebook/seamless-m4t-large')
print('✅ Model download completed!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Model downloaded and cached"
    else
        print_warning "Model download failed, but will be downloaded on first run"
    fi
fi

print_success "🎉 Quick setup complete!"
echo ""
echo "🚀 To run the service:"
echo "   1. Activate environment: source ~/seamless-env-quick.sh"  
echo "   2. Start service: python3 backend/main.py"
echo ""
echo "📱 Service will be available at: http://localhost:7860"