@echo off
REM Test WebSocket Audio Streaming Script
REM This script generates test audio and runs WebSocket tests

echo 🎵 WebSocket Audio Test Suite
echo =============================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM Install required packages if not present
echo 📦 Checking required packages...
python -c "import numpy, websockets" >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔧 Installing required packages...
    pip install numpy websockets
)

REM Generate test audio files if they don't exist
if not exist "samples\test_audio.wav" (
    echo 🎵 Generating test audio files...
    python generate_test_audio.py
    echo.
)

REM Show available test files
echo 📁 Available test files:
dir /b samples\*.wav 2>nul
echo.

REM Ask user which test to run
echo Choose a test:
echo 1. Quick test with short audio (test_audio.wav)
echo 2. Speech-like audio test (speech_like.wav) 
echo 3. Long streaming test (long_test.wav)
echo 4. Simple sine wave test (sine_440hz.wav)
echo 5. Custom file path
echo 6. Run all tests
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    set AUDIO_FILE=samples\test_audio.wav
    set TEST_NAME=Quick Test
) else if "%choice%"=="2" (
    set AUDIO_FILE=samples\speech_like.wav
    set TEST_NAME=Speech-like Test
) else if "%choice%"=="3" (
    set AUDIO_FILE=samples\long_test.wav
    set TEST_NAME=Long Streaming Test
) else if "%choice%"=="4" (
    set AUDIO_FILE=samples\sine_440hz.wav
    set TEST_NAME=Sine Wave Test
) else if "%choice%"=="5" (
    set /p AUDIO_FILE="Enter WAV file path: "
    set TEST_NAME=Custom File Test
) else if "%choice%"=="6" (
    goto run_all_tests
) else (
    echo ❌ Invalid choice. Using default test.
    set AUDIO_FILE=samples\test_audio.wav
    set TEST_NAME=Default Test
)

echo.
echo 🚀 Running %TEST_NAME% with %AUDIO_FILE%
echo Make sure your streaming server is running on localhost:8000
echo.
pause

python test_websocket_audio.py "%AUDIO_FILE%"
goto end

:run_all_tests
echo.
echo 🚀 Running all tests...
echo Make sure your streaming server is running on localhost:8000
echo.
pause

echo.
echo ======== Test 1: Quick Test ========
python test_websocket_audio.py samples\test_audio.wav
echo.

echo ======== Test 2: Speech-like Test ========
python test_websocket_audio.py samples\speech_like.wav  
echo.

echo ======== Test 3: Sine Wave Test ========
python test_websocket_audio.py samples\sine_440hz.wav
echo.

echo 🎉 All tests completed!

:end
echo.
echo 📊 Test Results:
if exist "received_audio_*.raw" (
    echo ✅ Audio responses received - check received_audio_*.raw files
    dir /b received_audio_*.raw 2>nul
) else (
    echo ⚠️ No audio responses found - check server logs
)

echo.
echo 💡 Tips:
echo - Check backend/streaming_server.py logs for detailed processing info
echo - Use audio software to play received_audio_*.raw files (16kHz, 16-bit, mono PCM)
echo - Try different chunk sizes with: python test_websocket_audio.py samples\test_audio.wav --chunk_size 512
echo.
pause