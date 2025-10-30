#!/bin/bash
# Complete log capture with output redirection

echo "🚀 Starting SeamlessStreaming server with COMPLETE output capture..."
echo "📁 ALL output (stdout + stderr) will be saved to complete_debug.log"

cd backend

# Use script command to capture everything or simple redirection
python -u streaming_server.py > complete_debug.log 2>&1 &
SERVER_PID=$!

echo "📊 Server started with PID: $SERVER_PID"
echo "💡 Run your test now from another terminal:"
echo "   python test_websocket_audio.py samples/harvard.wav --source_lang eng --target_lang ben --server wss://your-server:7860/ws/stream"
echo ""
echo "📋 To view logs in real-time: tail -f backend/complete_debug.log"
echo "⏹️  To stop server: kill $SERVER_PID"
echo ""

# Wait for user to stop
read -p "Press Enter when testing is complete to stop server and show logs..."

# Stop server
kill $SERVER_PID 2>/dev/null

echo ""
echo "🏁 Server stopped. Complete logs:"
echo "=================================="
cat backend/complete_debug.log
echo "=================================="
echo "📁 Full log saved in: backend/complete_debug.log"