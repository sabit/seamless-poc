#!/bin/bash

# Package installation checker for SeamlessStreaming Translation Service
# Checks which packages from requirements-native.txt are already installed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Check if requirements file exists
REQUIREMENTS_FILE="requirements-native.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_error "Requirements file '$REQUIREMENTS_FILE' not found"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found. Please install Python3 first."
    exit 1
fi

print_header "📦 Package Installation Checker"
print_header "================================"

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1)
print_status "Python version: $PYTHON_VERSION"

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    print_error "pip not found. Please install pip first."
    exit 1
fi

PIP_VERSION=$(python3 -m pip --version 2>&1)
print_status "Pip version: $PIP_VERSION"
echo ""

# Parse requirements and check each package
print_header "📋 Checking packages from $REQUIREMENTS_FILE:"
echo ""

# Counters
TOTAL_PACKAGES=0
INSTALLED_PACKAGES=0
MISSING_PACKAGES=0
VERSION_MISMATCHES=0

# Arrays to store results
INSTALLED_LIST=()
MISSING_LIST=()
MISMATCH_LIST=()

# Read and process requirements file
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    
    # Extract package name and version requirement
    PACKAGE_SPEC=$(echo "$line" | sed 's/[[:space:]]*#.*$//' | xargs)
    
    # Parse package name and version
    if [[ "$PACKAGE_SPEC" =~ ^([a-zA-Z0-9_-]+)(\[.*\])?(>=|==|<=|>|<|!=)(.+)$ ]]; then
        PACKAGE_NAME="${BASH_REMATCH[1]}"
        EXTRAS="${BASH_REMATCH[2]}"
        OPERATOR="${BASH_REMATCH[3]}"
        REQUIRED_VERSION="${BASH_REMATCH[4]}"
    elif [[ "$PACKAGE_SPEC" =~ ^([a-zA-Z0-9_-]+)(\[.*\])?$ ]]; then
        PACKAGE_NAME="${BASH_REMATCH[1]}"
        EXTRAS="${BASH_REMATCH[2]}"
        OPERATOR=""
        REQUIRED_VERSION=""
    else
        print_warning "Could not parse: $PACKAGE_SPEC"
        continue
    fi
    
    TOTAL_PACKAGES=$((TOTAL_PACKAGES + 1))
    
    # Check if package is installed
    INSTALLED_VERSION=$(python3 -c "
try:
    import pkg_resources
    try:
        version = pkg_resources.get_distribution('$PACKAGE_NAME').version
        print(version)
    except pkg_resources.DistributionNotFound:
        print('NOT_FOUND')
except ImportError:
    # Fallback method
    try:
        import $PACKAGE_NAME
        if hasattr($PACKAGE_NAME, '__version__'):
            print($PACKAGE_NAME.__version__)
        else:
            print('UNKNOWN_VERSION')
    except ImportError:
        print('NOT_FOUND')
" 2>/dev/null)
    
    # Display results
    if [ "$INSTALLED_VERSION" = "NOT_FOUND" ]; then
        print_error "❌ $PACKAGE_NAME - Not installed"
        MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
        MISSING_LIST+=("$PACKAGE_NAME")
    elif [ "$INSTALLED_VERSION" = "UNKNOWN_VERSION" ]; then
        print_warning "⚠️  $PACKAGE_NAME - Installed (version unknown)"
        INSTALLED_PACKAGES=$((INSTALLED_PACKAGES + 1))
        INSTALLED_LIST+=("$PACKAGE_NAME (version unknown)")
    else
        # Check version compatibility if specified
        if [ -n "$REQUIRED_VERSION" ]; then
            VERSION_CHECK=$(python3 -c "
from packaging import version
import sys
try:
    installed = version.parse('$INSTALLED_VERSION')
    required = version.parse('$REQUIRED_VERSION')
    
    if '$OPERATOR' == '>=':
        result = installed >= required
    elif '$OPERATOR' == '==':
        result = installed == required
    elif '$OPERATOR' == '<=':
        result = installed <= required
    elif '$OPERATOR' == '>':
        result = installed > required
    elif '$OPERATOR' == '<':
        result = installed < required
    elif '$OPERATOR' == '!=':
        result = installed != required
    else:
        result = True
    
    print('OK' if result else 'MISMATCH')
except Exception as e:
    print('ERROR')
" 2>/dev/null)
            
            if [ "$VERSION_CHECK" = "OK" ]; then
                print_success "✅ $PACKAGE_NAME - v$INSTALLED_VERSION (satisfies $OPERATOR$REQUIRED_VERSION)"
                INSTALLED_PACKAGES=$((INSTALLED_PACKAGES + 1))
                INSTALLED_LIST+=("$PACKAGE_NAME==$INSTALLED_VERSION")
            elif [ "$VERSION_CHECK" = "MISMATCH" ]; then
                print_warning "⚠️  $PACKAGE_NAME - v$INSTALLED_VERSION (requires $OPERATOR$REQUIRED_VERSION)"
                VERSION_MISMATCHES=$((VERSION_MISMATCHES + 1))
                MISMATCH_LIST+=("$PACKAGE_NAME: installed=$INSTALLED_VERSION, required=$OPERATOR$REQUIRED_VERSION")
            else
                print_warning "⚠️  $PACKAGE_NAME - v$INSTALLED_VERSION (could not verify version requirement)"
                INSTALLED_PACKAGES=$((INSTALLED_PACKAGES + 1))
                INSTALLED_LIST+=("$PACKAGE_NAME==$INSTALLED_VERSION")
            fi
        else
            print_success "✅ $PACKAGE_NAME - v$INSTALLED_VERSION"
            INSTALLED_PACKAGES=$((INSTALLED_PACKAGES + 1))
            INSTALLED_LIST+=("$PACKAGE_NAME==$INSTALLED_VERSION")
        fi
    fi
    
done < "$REQUIREMENTS_FILE"

# Summary
echo ""
print_header "📊 Summary:"
print_header "==========="
echo "Total packages checked: $TOTAL_PACKAGES"
echo "✅ Installed and compatible: $INSTALLED_PACKAGES"
echo "❌ Missing: $MISSING_PACKAGES"
echo "⚠️  Version mismatches: $VERSION_MISMATCHES"

# Show missing packages
if [ ${#MISSING_LIST[@]} -gt 0 ]; then
    echo ""
    print_header "📥 Missing packages:"
    for pkg in "${MISSING_LIST[@]}"; do
        echo "  - $pkg"
    done
fi

# Show version mismatches
if [ ${#MISMATCH_LIST[@]} -gt 0 ]; then
    echo ""
    print_header "⚠️  Version mismatches:"
    for mismatch in "${MISMATCH_LIST[@]}"; do
        echo "  - $mismatch"
    done
fi

# Show installed packages
if [ ${#INSTALLED_LIST[@]} -gt 0 ]; then
    echo ""
    print_header "✅ Installed packages:"
    for pkg in "${INSTALLED_LIST[@]}"; do
        echo "  - $pkg"
    done
fi

# Installation suggestions
echo ""
if [ $MISSING_PACKAGES -gt 0 ] || [ $VERSION_MISMATCHES -gt 0 ]; then
    print_header "💡 Suggestions:"
    
    if [ $MISSING_PACKAGES -gt 0 ]; then
        echo "To install missing packages:"
        echo "  pip install -r $REQUIREMENTS_FILE"
    fi
    
    if [ $VERSION_MISMATCHES -gt 0 ]; then
        echo "To upgrade packages with version mismatches:"
        echo "  pip install --upgrade -r $REQUIREMENTS_FILE"
    fi
    
    echo ""
    echo "Or use the setup script:"
    echo "  ./scripts/setup-native.sh"
else
    print_success "🎉 All packages are installed and compatible!"
fi

echo ""
print_header "🔧 Additional checks:"

# Check PyTorch CUDA
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    print_success "PyTorch CUDA check completed"
else
    print_warning "Could not verify PyTorch CUDA availability"
fi

# Check transformers
if python3 -c "from transformers import AutoProcessor; print('Transformers library working')" 2>/dev/null; then
    print_success "Transformers library check completed"
else
    print_warning "Could not verify transformers library"
fi