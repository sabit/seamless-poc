@echo off
echo Starting SeamlessStreaming server with detailed logging...
echo Logs will be saved to seamless_debug.log

REM Start the server in the backend directory
cd backend
python streaming_server.py

echo.
echo Server stopped. Check seamless_debug.log for detailed logs.
pause