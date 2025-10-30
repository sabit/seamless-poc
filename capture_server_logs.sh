#!/bin/bash
# Capture all logs from SeamlessStreaming server

echo "🚀 Starting SeamlessStreaming server with full logging capture..."
echo "📁 Logs will be saved to seamless_full_debug.log"

cd backend

# Capture both stdout and stderr to log file
python streaming_server.py 2>&1 | tee seamless_full_debug.log

echo ""
echo "📋 Server stopped. Full logs captured in seamless_full_debug.log"
echo "💡 You can now copy this file or view it with: cat seamless_full_debug.log"