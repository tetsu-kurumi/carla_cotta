#!/bin/bash
# Installation script for CARLA Python package

echo "=========================================="
echo "CARLA Python Package Installation"
echo "=========================================="

# Check current Python version
PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+')
echo "Current Python version: $PYTHON_VERSION"

# CARLA 0.9.16 supports Python 3.10, 3.11, 3.12
if [[ "$PYTHON_VERSION" == "3.10" || "$PYTHON_VERSION" == "3.11" || "$PYTHON_VERSION" == "3.12" ]]; then
    echo "✓ Python version compatible with CARLA"

    # Install the appropriate wheel
    WHEEL_FILE="/home/tetsuk/Downloads/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}-manylinux_2_31_x86_64.whl"

    echo "Installing CARLA from: $WHEEL_FILE"
    pip install "$WHEEL_FILE"

    # Test installation
    echo ""
    echo "Testing CARLA import..."
    python3 -c "import carla; print(f'✓ CARLA installed successfully: {carla.__file__}')"

else
    echo "✗ Python $PYTHON_VERSION is not supported by CARLA 0.9.16"
    echo ""
    echo "CARLA 0.9.16 supports: Python 3.10, 3.11, 3.12"
    echo "You are using: Python $PYTHON_VERSION"
    echo ""
    echo "Please create a new conda environment with Python 3.12:"
    echo ""
    echo "  conda create -n carla_cotta python=3.12"
    echo "  conda activate carla_cotta"
    echo "  pip install torch torchvision opencv-python scikit-learn matplotlib pandas"
    echo "  bash INSTALL_CARLA.sh"
    echo ""
    exit 1
fi
