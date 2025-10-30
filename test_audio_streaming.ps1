#!/usr/bin/env pwsh
# Test WebSocket Audio Streaming Script (PowerShell)
# This script generates test audio and runs WebSocket tests

Write-Host "🎵 WebSocket Audio Test Suite" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python and add it to PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install required packages if not present
Write-Host "📦 Checking required packages..." -ForegroundColor Yellow
try {
    python -c "import numpy, websockets" 2>$null
    Write-Host "✅ Required packages found" -ForegroundColor Green
} catch {
    Write-Host "🔧 Installing required packages..." -ForegroundColor Yellow
    pip install numpy websockets
}

# Generate test audio files if they don't exist
if (-not (Test-Path "samples/test_audio.wav")) {
    Write-Host "🎵 Generating test audio files..." -ForegroundColor Yellow
    python generate_test_audio.py
    Write-Host ""
}

# Show available test files
Write-Host "📁 Available test files:" -ForegroundColor Cyan
if (Test-Path "samples/*.wav") {
    Get-ChildItem "samples/*.wav" | Select-Object Name | Format-Table -HideTableHeaders
} else {
    Write-Host "⚠️ No WAV files found in samples directory" -ForegroundColor Yellow
}
Write-Host ""

# Menu for test selection
Write-Host "Choose a test:" -ForegroundColor Cyan
Write-Host "1. Quick test with short audio (test_audio.wav)"
Write-Host "2. Speech-like audio test (speech_like.wav)"
Write-Host "3. Long streaming test (long_test.wav)"
Write-Host "4. Simple sine wave test (sine_440hz.wav)"
Write-Host "5. Custom file path"
Write-Host "6. Run all tests"
Write-Host ""

$choice = Read-Host "Enter your choice (1-6)"

switch ($choice) {
    "1" {
        $audioFile = "samples/test_audio.wav"
        $testName = "Quick Test"
    }
    "2" {
        $audioFile = "samples/speech_like.wav"
        $testName = "Speech-like Test"
    }
    "3" {
        $audioFile = "samples/long_test.wav"
        $testName = "Long Streaming Test"
    }
    "4" {
        $audioFile = "samples/sine_440hz.wav"
        $testName = "Sine Wave Test"
    }
    "5" {
        $audioFile = Read-Host "Enter WAV file path"
        $testName = "Custom File Test"
    }
    "6" {
        # Run all tests
        Write-Host ""
        Write-Host "🚀 Running all tests..." -ForegroundColor Green
        Write-Host "Make sure your streaming server is running on localhost:8000" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to continue"
        
        $testFiles = @(
            @{File="samples/test_audio.wav"; Name="Quick Test"},
            @{File="samples/speech_like.wav"; Name="Speech-like Test"},
            @{File="samples/sine_440hz.wav"; Name="Sine Wave Test"}
        )
        
        foreach ($test in $testFiles) {
            Write-Host ""
            Write-Host "======== $($test.Name) ========" -ForegroundColor Cyan
            if (Test-Path $test.File) {
                python test_websocket_audio.py $test.File
            } else {
                Write-Host "⚠️ File not found: $($test.File)" -ForegroundColor Yellow
            }
            Write-Host ""
        }
        
        Write-Host "🎉 All tests completed!" -ForegroundColor Green
        
        # Show results
        Write-Host ""
        Write-Host "📊 Test Results:" -ForegroundColor Cyan
        $audioFiles = Get-ChildItem "received_audio_*.raw" -ErrorAction SilentlyContinue
        if ($audioFiles) {
            Write-Host "✅ Audio responses received:" -ForegroundColor Green
            $audioFiles | Select-Object Name, Length | Format-Table
        } else {
            Write-Host "⚠️ No audio responses found - check server logs" -ForegroundColor Yellow
        }
        
        Write-Host ""
        Write-Host "💡 Tips:" -ForegroundColor Cyan
        Write-Host "- Check backend/streaming_server.py logs for detailed processing info"
        Write-Host "- Use audio software to play received_audio_*.raw files (16kHz, 16-bit, mono PCM)"
        Write-Host "- Try different chunk sizes with: python test_websocket_audio.py samples/test_audio.wav --chunk_size 512"
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit
    }
    default {
        Write-Host "❌ Invalid choice. Using default test." -ForegroundColor Red
        $audioFile = "samples/test_audio.wav"
        $testName = "Default Test"
    }
}

# Run single test
Write-Host ""
Write-Host "🚀 Running $testName with $audioFile" -ForegroundColor Green
Write-Host "Make sure your streaming server is running on localhost:8000" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to continue"

if (Test-Path $audioFile) {
    python test_websocket_audio.py $audioFile
} else {
    Write-Host "❌ File not found: $audioFile" -ForegroundColor Red
    exit 1
}

# Show results
Write-Host ""
Write-Host "📊 Test Results:" -ForegroundColor Cyan
$audioFiles = Get-ChildItem "received_audio_*.raw" -ErrorAction SilentlyContinue
if ($audioFiles) {
    Write-Host "✅ Audio responses received:" -ForegroundColor Green
    $audioFiles | Select-Object Name, Length | Format-Table
} else {
    Write-Host "⚠️ No audio responses found - check server logs" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Cyan
Write-Host "- Check backend/streaming_server.py logs for detailed processing info"
Write-Host "- Use audio software to play received_audio_*.raw files (16kHz, 16-bit, mono PCM)"
Write-Host "- Try different chunk sizes with: python test_websocket_audio.py samples/test_audio.wav --chunk_size 512"
Write-Host ""
Read-Host "Press Enter to exit"