#!/bin/bash
# CVAT Integration Installation Script for Linux/Mac
# This script installs all dependencies needed for CVAT integration

set -e  # Exit on error

echo "========================================"
echo "CVAT Integration - Installation Script"
echo "========================================"
echo

# Check Python version
echo "[1/4] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3.8+ required. Please upgrade Python."
    exit 1
fi
echo "     ✓ Python version is compatible"

echo
echo "[2/4] Installing CVAT dependencies..."
pip3 install -r requirements_cvat.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "     ✓ Dependencies installed"

echo
echo "[3/4] Verifying CVAT SDK installation..."
python3 -c "import cvat_sdk; print('     ✓ CVAT SDK installed')"
if [ $? -ne 0 ]; then
    echo "ERROR: CVAT SDK verification failed"
    exit 1
fi

echo
echo "[4/4] Verifying integration package..."
python3 -c "from cvat_integration import format_converter; print('     ✓ Integration package ready')"
if [ $? -ne 0 ]; then
    echo "ERROR: Integration package verification failed"
    exit 1
fi

echo
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Start CVAT server (see CVAT_SETUP.md)"
echo "2. Configure cvat_config.json with your credentials"
echo "3. Run your first workflow (see CVAT_QUICKSTART.md)"
echo
echo "Documentation:"
echo "- CVAT_SETUP.md       : Detailed setup guide"
echo "- CVAT_QUICKSTART.md  : Quick start examples"
echo "- README_CVAT_INTEGRATION.md : Complete reference"
echo
