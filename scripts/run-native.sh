#!/bin/bash

# Quick start script for native VM deployment
# Run this after setup-native.sh

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

# Check if environment setup exists
if [ ! -f "$HOME/seamless-env.sh" ]; then
    print_error "Environment not set up. Run ./scripts/setup-native.sh first"
    exit 1
fi

# Activate environment
print_status "Activating SeamlessStreaming environment..."
source "$HOME/seamless-env.sh"

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    print_error "backend/main.py not found. Please run from the seamless-poc directory"
    exit 1
fi

# Check GPU availability
print_status "Checking GPU availability..."
GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$GPU_AVAILABLE" = "True" ]; then
    print_success "GPU is available for acceleration"
else
    print_warning "GPU not available, will use CPU (slower)"
fi

# Get VM external IP
EXTERNAL_IP=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/external-ip 2>/dev/null || echo "localhost")

print_success "🚀 Starting SeamlessStreaming Translation Service..."
echo ""
echo "📱 Access the service at:"
echo "   Local:    http://localhost:7860"
echo "   External: http://$EXTERNAL_IP:7860"
echo ""
echo "⏹️  To stop: Press Ctrl+C"
echo ""

# Start the service
python3 backend/main.py