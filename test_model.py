#!/usr/bin/env python3
"""
Test script to verify SeamlessM4T model availability and imports
Run this to test if the model can be loaded before building the Docker image
"""

import sys
import torch

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        from transformers import SeamlessM4TModel, AutoProcessor
        print("✅ SeamlessM4TModel and AutoProcessor imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        import librosa
        import soundfile
        print("✅ Audio libraries imported successfully")
    except ImportError as e:
        print(f"❌ Audio import error: {e}")
        return False
    
    return True

def test_model_availability():
    """Test if the model can be loaded"""
    print("\nTesting model availability...")
    
    try:
        from transformers import SeamlessM4TModel, AutoProcessor
        
        # Test processor first (smaller download)
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-large")
        print("✅ Processor loaded successfully")
        
        # Test model loading (this will download if not cached)
        print("Loading model... (this may take a while on first run)")
        model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-large")
        print(f"✅ Model loaded successfully")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Model type: {type(model)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        device_props = torch.cuda.get_device_properties(0)
        
        print(f"✅ CUDA available: {device_name}")
        print(f"   CUDA version: {cuda_version}")
        print(f"   GPU memory: {device_props.total_memory / 1e9:.1f} GB")
        print(f"   Compute capability: {device_props.major}.{device_props.minor}")
        
        # Check if it's a T4 GPU
        if "T4" in device_name:
            print("✅ NVIDIA T4 detected - optimized for inference")
            
        # Check CUDA version compatibility
        if cuda_version and cuda_version.startswith("12"):
            print("✅ CUDA 12.x detected - excellent compatibility")
        elif cuda_version and cuda_version.startswith("11"):
            print("⚠️  CUDA 11.x detected - should work but 12.x is recommended")
        
        return True
    else:
        print("⚠️  CUDA not available, will use CPU")
        return False

def main():
    """Main test function"""
    print("🧪 SeamlessM4T Model Test Suite")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install required packages:")
        print("pip install transformers torch torchaudio librosa soundfile")
        sys.exit(1)
    
    # Test GPU
    gpu_available = test_gpu_availability()
    
    # Test model loading
    if not test_model_availability():
        print("\n❌ Model loading test failed.")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("✅ Ready to build Docker image")
    
    if gpu_available:
        print("✅ GPU acceleration will be available")
    else:
        print("⚠️  Will run on CPU only (slower inference)")

if __name__ == "__main__":
    main()