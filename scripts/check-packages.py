#!/usr/bin/env python3
"""
Package installation checker for SeamlessStreaming Translation Service
Checks which packages from requirements-native.txt are already installed
"""

import sys
import subprocess
import re
import os
from pathlib import Path

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_colored(text, color):
    print(f"{color}{text}{Colors.NC}")

def print_status(text):
    print_colored(f"[INFO] {text}", Colors.BLUE)

def print_success(text):
    print_colored(f"[SUCCESS] {text}", Colors.GREEN)

def print_warning(text):
    print_colored(f"[WARNING] {text}", Colors.YELLOW)

def print_error(text):
    print_colored(f"[ERROR] {text}", Colors.RED)

def print_header(text):
    print_colored(text, Colors.CYAN)

def get_installed_version(package_name):
    """Get the installed version of a package"""
    try:
        import pkg_resources
        try:
            version = pkg_resources.get_distribution(package_name).version
            return version
        except pkg_resources.DistributionNotFound:
            return None
    except ImportError:
        # Fallback method
        try:
            module = __import__(package_name.replace('-', '_'))
            if hasattr(module, '__version__'):
                return module.__version__
            else:
                return "UNKNOWN_VERSION"
        except ImportError:
            return None

def parse_requirement(line):
    """Parse a requirement line and extract package name, operator, and version"""
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None, None, None
    
    # Remove inline comments
    line = re.sub(r'\s*#.*$', '', line)
    
    # Parse package specification
    # Handle extras like package[extra1,extra2]>=1.0.0
    match = re.match(r'^([a-zA-Z0-9_-]+)(\[.*\])?(>=|==|<=|>|<|!=)(.+)$', line)
    if match:
        package_name = match.group(1)
        operator = match.group(3)
        version = match.group(4)
        return package_name, operator, version
    
    # Handle simple package name without version
    match = re.match(r'^([a-zA-Z0-9_-]+)(\[.*\])?$', line)
    if match:
        package_name = match.group(1)
        return package_name, None, None
    
    return None, None, None

def compare_versions(installed, operator, required):
    """Compare installed version with required version"""
    try:
        from packaging import version
        
        installed_ver = version.parse(installed)
        required_ver = version.parse(required)
        
        if operator == '>=':
            return installed_ver >= required_ver
        elif operator == '==':
            return installed_ver == required_ver
        elif operator == '<=':
            return installed_ver <= required_ver
        elif operator == '>':
            return installed_ver > required_ver
        elif operator == '<':
            return installed_ver < required_ver
        elif operator == '!=':
            return installed_ver != required_ver
        else:
            return True
    except Exception:
        return None

def check_special_functionality():
    """Check if special functionality works"""
    results = []
    
    # Check PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            results.append(("PyTorch CUDA", f"✅ Available - {gpu_name}"))
        else:
            results.append(("PyTorch CUDA", "⚠️  Not available (CPU only)"))
    except ImportError:
        results.append(("PyTorch CUDA", "❌ PyTorch not installed"))
    except Exception as e:
        results.append(("PyTorch CUDA", f"⚠️  Error: {e}"))
    
    # Check Transformers
    try:
        from transformers import AutoProcessor
        results.append(("Transformers", "✅ Working"))
    except ImportError:
        results.append(("Transformers", "❌ Not installed"))
    except Exception as e:
        results.append(("Transformers", f"⚠️  Error: {e}"))
    
    # Check audio libraries
    try:
        import librosa
        import soundfile
        results.append(("Audio libraries", "✅ librosa and soundfile available"))
    except ImportError as e:
        results.append(("Audio libraries", f"⚠️  Missing: {e}"))
    
    return results

def main():
    requirements_file = "requirements-native.txt"
    
    print_header("📦 Package Installation Checker")
    print_header("================================")
    
    # Check if requirements file exists
    if not os.path.exists(requirements_file):
        print_error(f"Requirements file '{requirements_file}' not found")
        sys.exit(1)
    
    # Print Python version
    python_version = sys.version.split()[0]
    print_status(f"Python version: {python_version}")
    
    # Check pip
    try:
        pip_version = subprocess.check_output([sys.executable, '-m', 'pip', '--version'], 
                                             stderr=subprocess.STDOUT, text=True).strip()
        print_status(f"Pip version: {pip_version}")
    except subprocess.CalledProcessError:
        print_error("pip not available")
        sys.exit(1)
    
    print()
    print_header(f"📋 Checking packages from {requirements_file}:")
    print()
    
    # Counters
    total_packages = 0
    installed_packages = 0
    missing_packages = 0
    version_mismatches = 0
    
    # Lists to store results
    installed_list = []
    missing_list = []
    mismatch_list = []
    
    # Read and process requirements file
    with open(requirements_file, 'r') as f:
        for line in f:
            package_name, operator, required_version = parse_requirement(line)
            
            if not package_name:
                continue
            
            total_packages += 1
            
            # Check if package is installed
            installed_version = get_installed_version(package_name)
            
            if installed_version is None:
                print_error(f"❌ {package_name} - Not installed")
                missing_packages += 1
                missing_list.append(package_name)
            elif installed_version == "UNKNOWN_VERSION":
                print_warning(f"⚠️  {package_name} - Installed (version unknown)")
                installed_packages += 1
                installed_list.append(f"{package_name} (version unknown)")
            else:
                # Check version compatibility if specified
                if operator and required_version:
                    compatible = compare_versions(installed_version, operator, required_version)
                    
                    if compatible is True:
                        print_success(f"✅ {package_name} - v{installed_version} (satisfies {operator}{required_version})")
                        installed_packages += 1
                        installed_list.append(f"{package_name}=={installed_version}")
                    elif compatible is False:
                        print_warning(f"⚠️  {package_name} - v{installed_version} (requires {operator}{required_version})")
                        version_mismatches += 1
                        mismatch_list.append(f"{package_name}: installed={installed_version}, required={operator}{required_version}")
                    else:
                        print_warning(f"⚠️  {package_name} - v{installed_version} (could not verify version requirement)")
                        installed_packages += 1
                        installed_list.append(f"{package_name}=={installed_version}")
                else:
                    print_success(f"✅ {package_name} - v{installed_version}")
                    installed_packages += 1
                    installed_list.append(f"{package_name}=={installed_version}")
    
    # Summary
    print()
    print_header("📊 Summary:")
    print_header("===========")
    print(f"Total packages checked: {total_packages}")
    print(f"✅ Installed and compatible: {installed_packages}")
    print(f"❌ Missing: {missing_packages}")
    print(f"⚠️  Version mismatches: {version_mismatches}")
    
    # Show missing packages
    if missing_list:
        print()
        print_header("📥 Missing packages:")
        for pkg in missing_list:
            print(f"  - {pkg}")
    
    # Show version mismatches
    if mismatch_list:
        print()
        print_header("⚠️  Version mismatches:")
        for mismatch in mismatch_list:
            print(f"  - {mismatch}")
    
    # Show installed packages
    if installed_list:
        print()
        print_header("✅ Installed packages:")
        for pkg in installed_list:
            print(f"  - {pkg}")
    
    # Installation suggestions
    print()
    if missing_packages > 0 or version_mismatches > 0:
        print_header("💡 Suggestions:")
        
        if missing_packages > 0:
            print("To install missing packages:")
            print(f"  pip install -r {requirements_file}")
        
        if version_mismatches > 0:
            print("To upgrade packages with version mismatches:")
            print(f"  pip install --upgrade -r {requirements_file}")
        
        print()
        print("Or use the setup script:")
        print("  ./scripts/setup-native.sh")
    else:
        print_success("🎉 All packages are installed and compatible!")
    
    # Additional functionality checks
    print()
    print_header("🔧 Additional functionality checks:")
    
    special_checks = check_special_functionality()
    for check_name, result in special_checks:
        print(f"{check_name}: {result}")
    
    print()
    if missing_packages == 0 and version_mismatches == 0:
        print_success("🚀 Ready to run SeamlessStreaming Translation Service!")
    else:
        print_warning("⚠️  Some packages need attention before running the service")

if __name__ == "__main__":
    main()