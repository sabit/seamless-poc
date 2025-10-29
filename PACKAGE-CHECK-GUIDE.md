# Package Checker Usage Guide

## Overview
Two scripts are available to check which packages from `requirements-native.txt` are already installed:

## Scripts

### 1. Bash Version: `check-packages.sh`
**Best for Linux/Mac/WSL environments**

```bash
# Make executable and run
chmod +x scripts/check-packages.sh
./scripts/check-packages.sh
```

**Features:**
- Color-coded output
- Detailed version checking
- PyTorch CUDA verification
- Installation suggestions
- Works with bash/zsh shells

### 2. Python Version: `check-packages.py`
**Best for cross-platform use (Windows/Linux/Mac)**

```bash
# Run with Python 3
python3 scripts/check-packages.py

# Or on Windows
python scripts/check-packages.py
```

**Features:**
- Pure Python (no shell dependencies)
- Same functionality as bash version
- Works on any platform with Python 3
- Detailed package analysis

## What the Scripts Check

### Package Status
- ✅ **Installed and Compatible** - Package is installed with correct version
- ❌ **Missing** - Package is not installed
- ⚠️ **Version Mismatch** - Package installed but wrong version
- ⚠️ **Unknown Version** - Package installed but version cannot be determined

### Additional Checks
- **PyTorch CUDA** - GPU acceleration availability
- **Transformers** - Model loading capability
- **Audio Libraries** - librosa and soundfile functionality

## Example Output

```
📦 Package Installation Checker
================================
[INFO] Python version: 3.10.12
[INFO] Pip version: pip 23.2.1

📋 Checking packages from requirements-native.txt:

✅ fastapi - v0.104.1 (satisfies ==0.104.1)
✅ uvicorn - v0.24.0
❌ torch - Not installed
⚠️  transformers - v4.35.0 (requires ==4.37.0)

📊 Summary:
===========
Total packages checked: 12
✅ Installed and compatible: 8
❌ Missing: 2
⚠️ Version mismatches: 2

💡 Suggestions:
To install missing packages:
  pip install -r requirements-native.txt

🔧 Additional functionality checks:
PyTorch CUDA: ❌ PyTorch not installed
Transformers: ⚠️ Version mismatch
Audio libraries: ✅ librosa and soundfile available
```

## Quick Commands

```bash
# Check current environment
python3 scripts/check-packages.py

# Install missing packages
pip install -r requirements-native.txt

# Use full setup script instead
./scripts/setup-native.sh
```

## Troubleshooting

### Permission Denied (Linux/Mac)
```bash
chmod +x scripts/check-packages.sh
```

### Python Not Found
```bash
# Try different Python commands
python3 scripts/check-packages.py
python scripts/check-packages.py
py scripts/check-packages.py  # Windows
```

### Missing packaging Module
```bash
pip install packaging
```

The scripts help you understand your current environment before running the full setup process!