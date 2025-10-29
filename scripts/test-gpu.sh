#!/bin/bash

# Simple CUDA and Docker GPU test script
# This script tests basic GPU functionality without relying on specific CUDA image versions

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

echo "🧪 CUDA and Docker GPU Test"
echo "=========================="

# Test 1: Check if nvidia-smi works
print_status "Testing nvidia-smi..."
if nvidia-smi > /dev/null 2>&1; then
    print_success "nvidia-smi is working"
    nvidia-smi --query-gpu=name,memory.total,cuda_version --format=csv,noheader
else
    print_error "nvidia-smi failed. CUDA drivers may not be installed."
    exit 1
fi

# Test 2: Check Docker
print_status "Testing Docker..."
if docker --version > /dev/null 2>&1; then
    print_success "Docker is installed: $(docker --version)"
else
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Test 3: Check Docker daemon
print_status "Testing Docker daemon..."
if docker info > /dev/null 2>&1; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running"
    exit 1
fi

# Test 4: Test Docker GPU access
print_status "Testing Docker GPU access..."
if docker run --rm --gpus all ubuntu:20.04 nvidia-smi > /dev/null 2>&1; then
    print_success "Docker can access GPU and nvidia-smi works in containers"
elif docker run --rm --gpus all alpine:latest echo "GPU access test" > /dev/null 2>&1; then
    print_success "Docker --gpus flag is working"
    print_status "nvidia-smi not available in test container, but GPU access is configured"
else
    print_error "Docker cannot access GPU. nvidia-docker2 may not be installed."
    print_status "Install with: sudo apt-get update && sudo apt-get install -y nvidia-docker2"
    print_status "Then restart: sudo systemctl restart docker"
    exit 1
fi

# Test 5: Test if we can access NVIDIA container registry
print_status "Testing access to NVIDIA Container Registry..."
if docker pull --quiet nvcr.io/nvidia/pytorch:23.08-py3 > /dev/null 2>&1; then
    print_success "Can access NVIDIA PyTorch base images"
    docker rmi nvcr.io/nvidia/pytorch:23.08-py3 > /dev/null 2>&1 || true
else
    print_warning "Cannot access NVIDIA Container Registry. Check internet connection."
    print_status "The build will attempt to pull images during Docker build process"
fi

print_success "🎉 All GPU tests passed!"
echo ""
echo "✅ Ready to build SeamlessStreaming Translation Service"
echo "🚀 Run: ./scripts/build-and-run.sh"