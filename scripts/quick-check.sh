#!/bin/bash

# Quick pre-build check script
# Tests basic requirements without downloading large images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🔍 Quick Pre-Build Check"
echo "======================="

# Check 1: nvidia-smi
print_status "Checking NVIDIA drivers..."
if nvidia-smi > /dev/null 2>&1; then
    print_success "NVIDIA drivers are working"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "   GPU: $GPU_INFO"
else
    print_error "nvidia-smi failed - NVIDIA drivers not properly installed"
    exit 1
fi

# Check 2: Docker
print_status "Checking Docker..."
if docker --version > /dev/null 2>&1 && docker info > /dev/null 2>&1; then
    print_success "Docker is working"
else
    print_error "Docker is not working properly"
    exit 1
fi

# Check 3: Docker GPU flag (simple test)
print_status "Testing Docker GPU support..."
if docker run --rm --gpus all alpine:latest echo "GPU test successful" > /dev/null 2>&1; then
    print_success "Docker --gpus flag is working"
else
    print_error "Docker --gpus flag failed - nvidia-docker2 may not be installed"
    echo "Install with: sudo apt install nvidia-docker2 && sudo systemctl restart docker"
    exit 1
fi

print_success "✅ All basic checks passed!"
echo ""
echo "🚀 Ready to build! Run: ./scripts/build-and-run.sh"