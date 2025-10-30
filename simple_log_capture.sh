#!/bin/bash
# Simple complete output capture - run from project root

echo "🚀 Starting server with complete output redirection..."
echo "📂 Running from project root directory"

# Just redirect everything to a file (from project root)
python -u backend/streaming_server.py 2>&1 | tee complete_output.log

echo ""
echo "📁 Complete output saved to: complete_output.log"