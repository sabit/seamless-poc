#!/bin/bash

# Build and run script for SeamlessStreaming Translation Service
# Run this script on your GCP VM to build and deploy the application

set -e

echo "🚀 Building SeamlessStreaming Translation Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="seamless-translator"
CONTAINER_NAME="seamless-translator-app"
VOLUME_NAME="seamless-model-cache"
PORT=7860

# Function to print colored output
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

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker service."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    print_warning "NVIDIA Docker runtime not available. GPU acceleration may not work."
fi

print_status "Stopping existing container if running..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

print_status "Creating Docker volume for model cache..."
docker volume create $VOLUME_NAME 2>/dev/null || print_status "Volume already exists"

print_status "Building Docker image..."
docker build -t $IMAGE_NAME . --no-cache

if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    exit 1
fi

print_success "Docker image built successfully!"

print_status "Starting container with persistent model cache..."
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    -p $PORT:$PORT \
    --restart unless-stopped \
    -v $VOLUME_NAME:/app/model_cache \
    $IMAGE_NAME

if [ $? -ne 0 ]; then
    print_error "Failed to start container!"
    exit 1
fi

print_success "Container started successfully!"

# Wait for the service to be ready
print_status "Waiting for service to be ready..."
for i in {1..60}; do
    if curl -f http://localhost:$PORT/health >/dev/null 2>&1; then
        print_success "Service is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        print_error "Service failed to start within 60 seconds"
        print_status "Checking container logs..."
        docker logs $CONTAINER_NAME --tail 50
        exit 1
    fi
    sleep 1
done

# Get VM external IP (if on GCP)
EXTERNAL_IP=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/external-ip 2>/dev/null || echo "localhost")

print_success "🎉 SeamlessStreaming Translation Service is now running!"
echo ""
echo "📱 Access the web interface at:"
echo "   http://$EXTERNAL_IP:$PORT"
echo ""
echo "🔧 Container management commands:"
echo "   View logs: docker logs -f $CONTAINER_NAME"
echo "   Stop:      docker stop $CONTAINER_NAME"
echo "   Start:     docker start $CONTAINER_NAME"
echo "   Remove:    docker rm $CONTAINER_NAME"
echo ""
echo "📦 Volume management:"
echo "   Inspect:   docker volume inspect $VOLUME_NAME"
echo "   Remove:    docker volume rm $VOLUME_NAME"
echo "   Size:      docker system df -v"
echo ""
echo "📊 System status:"
docker ps --filter name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"