#!/usr/bin/env python3
"""
Compatibility checker for SeamlessStreaming with gold image packages
Tests if the pre-installed versions are compatible with our application
"""

import sys
import traceback

def test_fastapi_compatibility():
    """Test FastAPI features we use"""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        # Test basic app creation
        app = FastAPI(title="Test")
        
        print("✅ FastAPI: Compatible")
        return True
    except Exception as e:
        print(f"❌ FastAPI: Incompatible - {e}")
        return False

def test_uvicorn_compatibility():
    """Test Uvicorn features we use"""
    try:
        import uvicorn
        
        # Check if we can create a basic config
        config = uvicorn.Config(
            "main:app",
            host="0.0.0.0",
            port=7860,
            log_level="info"
        )
        
        print("✅ Uvicorn: Compatible")
        return True
    except Exception as e:
        print(f"❌ Uvicorn: Incompatible - {e}")
        return False

def test_websockets_compatibility():
    """Test WebSockets features we use"""
    try:
        import websockets
        
        # Basic import test - we use it through FastAPI
        print("✅ WebSockets: Compatible")
        return True
    except Exception as e:
        print(f"❌ WebSockets: Incompatible - {e}")
        return False

def test_aiofiles_compatibility():
    """Test aiofiles features we use"""
    try:
        import aiofiles
        
        print("✅ aiofiles: Compatible")
        return True
    except Exception as e:
        print(f"❌ aiofiles: Incompatible - {e}")
        return False

def test_torch_compatibility():
    """Test PyTorch features we use"""
    try:
        import torch
        import torchaudio
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ PyTorch + CUDA: Compatible - {device_name}")
        else:
            print("⚠️ PyTorch: Compatible but no CUDA (will use CPU)")
        
        # Test torchaudio transforms we use
        resampler = torchaudio.transforms.Resample(16000, 22050)
        print("✅ TorchAudio: Compatible")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch/TorchAudio: Incompatible - {e}")
        return False

def test_numpy_scipy_compatibility():
    """Test NumPy/SciPy features we use"""
    try:
        import numpy as np
        import scipy
        
        # Test basic operations we use
        test_array = np.array([1, 2, 3])
        normalized = test_array / np.max(test_array)
        
        print("✅ NumPy: Compatible")
        print("✅ SciPy: Compatible")
        return True
    except Exception as e:
        print(f"❌ NumPy/SciPy: Incompatible - {e}")
        return False

def test_missing_packages():
    """Check if required packages are missing"""
    missing_packages = []
    
    packages_to_check = [
        'transformers',
        'librosa', 
        'soundfile',
        'sentencepiece',
        'accelerate',
        'huggingface_hub'
    ]
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    return missing_packages

def test_transformers_functionality():
    """Test transformers functionality if available"""
    try:
        from transformers import SeamlessM4TModel, AutoProcessor
        print("✅ Transformers: SeamlessM4T classes available")
        return True
    except ImportError:
        print("ℹ️ Transformers: Not installed yet (expected)")
        return False
    except Exception as e:
        print(f"⚠️ Transformers: Installed but may have issues - {e}")
        return False

def main():
    print("🔍 Gold Image Compatibility Checker")
    print("==================================")
    print()
    
    print("Testing pre-installed packages:")
    print("-" * 30)
    
    compatibility_results = []
    
    # Test pre-installed packages
    compatibility_results.append(test_fastapi_compatibility())
    compatibility_results.append(test_uvicorn_compatibility()) 
    compatibility_results.append(test_websockets_compatibility())
    compatibility_results.append(test_aiofiles_compatibility())
    compatibility_results.append(test_torch_compatibility())
    compatibility_results.append(test_numpy_scipy_compatibility())
    
    print()
    print("Checking for missing packages:")
    print("-" * 30)
    
    missing_packages = test_missing_packages()
    test_transformers_functionality()
    
    print()
    print("Summary:")
    print("=" * 30)
    
    compatible_count = sum(compatibility_results)
    total_preinstalled = len(compatibility_results)
    
    print(f"Pre-installed packages compatible: {compatible_count}/{total_preinstalled}")
    print(f"Missing packages: {len(missing_packages)}")
    
    if missing_packages:
        print()
        print("To install missing packages:")
        print("pip install " + " ".join(missing_packages))
        print()
        print("Or use the setup script:")
        print("./scripts/setup-native.sh")
    
    print()
    if compatible_count == total_preinstalled and len(missing_packages) <= 8:
        print("✅ Gold image is compatible! You can proceed with minimal installation.")
    else:
        print("⚠️ Some compatibility issues detected. Check the details above.")
    
    # Show package versions for reference
    print()
    print("Current package versions:")
    print("-" * 25)
    
    packages_to_show = ['fastapi', 'uvicorn', 'websockets', 'aiofiles', 'torch', 'numpy', 'scipy']
    
    for package in packages_to_show:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: not installed")

if __name__ == "__main__":
    main()