#!/bin/bash

# Stop and cleanup script for SeamlessStreaming Translation Service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONTAINER_NAME="seamless-translator-app"
IMAGE_NAME="seamless-translator"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Stopping SeamlessStreaming Translation Service..."

# Stop container
if docker ps -q --filter name=$CONTAINER_NAME | grep -q .; then
    print_status "Stopping container..."
    docker stop $CONTAINER_NAME
    print_success "Container stopped"
else
    print_status "Container is not running"
fi

# Remove container
if docker ps -aq --filter name=$CONTAINER_NAME | grep -q .; then
    print_status "Removing container..."
    docker rm $CONTAINER_NAME
    print_success "Container removed"
fi

# Optionally remove image (uncomment if needed)
# print_status "Removing Docker image..."
# docker rmi $IMAGE_NAME 2>/dev/null || print_status "Image already removed or in use"

print_success "🛑 SeamlessStreaming Translation Service stopped and cleaned up!"

# Show remaining Docker resources
echo ""
echo "📊 Remaining containers:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "💿 Docker images:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"