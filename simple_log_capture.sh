#!/bin/bash
# Simple complete output capture

echo "🚀 Starting server with complete output redirection..."

cd backend

# Just redirect everything to a file
python -u streaming_server.py 2>&1 | tee complete_output.log

echo ""
echo "📁 Complete output saved to: backend/complete_output.log"