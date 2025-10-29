#!/bin/bash

# Quick run script for SeamlessStreaming service
# Uses gold image packages with minimal additions

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m' 
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    print_warning "Not in seamless-poc directory, navigating..."
    cd /path/to/seamless-poc  # User should update this
fi

print_status "🚀 Starting SeamlessStreaming Service"

# Activate environment if available
if [ -f "$HOME/seamless-env-quick.sh" ]; then
    source "$HOME/seamless-env-quick.sh"
else
    # Minimal environment setup
    export PYTHONPATH="$PWD"
    export TRANSFORMERS_CACHE="$HOME/seamless-model-cache"
    export HF_HOME="$HOME/seamless-model-cache"
    export CUDA_VISIBLE_DEVICES=0
    
    mkdir -p "$HOME/seamless-model-cache"
fi

# Check GPU availability
print_status "Checking GPU..."
GPU_CHECK=$(python3 -c "import torch; print('✅ GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "⚠️ GPU check failed")
echo "$GPU_CHECK"

# Start the service
print_success "Starting service on http://localhost:7860..."
print_status "Frontend will be available at http://localhost:7860"
print_status "Use Ctrl+C to stop"
echo ""

# Run with uvicorn using gold image version
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 7860 --reload