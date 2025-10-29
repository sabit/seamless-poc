#!/bin/bash

# Development mode script - automatically restarts service on code changes
# Useful for development and testing

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
print_status "Activating development environment..."
source "$HOME/seamless-env.sh"

# Install development dependencies
print_status "Installing development dependencies..."
pip install watchdog[watchmedo]

# Get VM external IP
EXTERNAL_IP=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/external-ip 2>/dev/null || echo "localhost")

print_success "🔧 Development mode started!"
echo ""
echo "📱 Access the service at:"
echo "   Local:    http://localhost:7860"
echo "   External: http://$EXTERNAL_IP:7860"
echo ""
echo "🔄 Auto-restart enabled - service will restart when you change files"
echo "⏹️  To stop: Press Ctrl+C"
echo ""

# Start with auto-reload
uvicorn backend.main:app --host 0.0.0.0 --port 7860 --reload