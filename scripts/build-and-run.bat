@echo off
REM Build and run script for SeamlessStreaming Translation Service (Windows version)
REM Run this script on your Windows machine if testing locally

setlocal enabledelayedexpansion

echo 🚀 Building SeamlessStreaming Translation Service...

set IMAGE_NAME=seamless-translator
set CONTAINER_NAME=seamless-translator-app
set PORT=7860

REM Check if Docker is installed and running
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    exit /b 1
)

echo [INFO] Stopping existing container if running...
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1

echo [INFO] Building Docker image...
docker build -t %IMAGE_NAME% . --no-cache

if errorlevel 1 (
    echo [ERROR] Docker build failed!
    exit /b 1
)

echo [SUCCESS] Docker image built successfully!

echo [INFO] Starting container...
docker run -d --name %CONTAINER_NAME% --gpus all -p %PORT%:%PORT% --restart unless-stopped -v "%cd%\cache:/app/cache" %IMAGE_NAME%

if errorlevel 1 (
    echo [ERROR] Failed to start container!
    exit /b 1
)

echo [SUCCESS] Container started successfully!

echo [INFO] Waiting for service to be ready...
timeout /t 10 /nobreak >nul

echo 🎉 SeamlessStreaming Translation Service is now running!
echo.
echo 📱 Access the web interface at:
echo    http://localhost:%PORT%
echo.
echo 🔧 Container management commands:
echo    View logs: docker logs -f %CONTAINER_NAME%
echo    Stop:      docker stop %CONTAINER_NAME%
echo    Start:     docker start %CONTAINER_NAME%
echo    Remove:    docker rm %CONTAINER_NAME%
echo.

docker ps --filter name=%CONTAINER_NAME%

endlocal