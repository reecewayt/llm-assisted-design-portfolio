#!/bin/bash
# Simple script to set up the Python environment for CNN benchmarking

# Determine script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Flag to track if this is a new environment
FRESH_INSTALL=false

# Check if virtual environment already exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Activating it..."
else
    # Create a virtual environment
    echo "Creating Python virtual environment..."
    python -m venv .venv
    FRESH_INSTALL=true
fi

# Activate the virtual environment
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    .venv\Scripts\activate
fi

# Check if activation was successful
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment activation failed."
    exit 1
fi

# Install required packages only for fresh install
if [ "$FRESH_INSTALL" = true ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
    echo "Setup complete! Your Python environment is ready."
else
    echo "Virtual environment activated. Use --update flag to update packages."

    # Check if --update flag is provided
    if [[ "$1" == "--update" ]]; then
        echo "Updating packages from requirements.txt..."
        pip install -r requirements.txt
    fi
fi

echo "To activate the virtual environment in the future:"
echo "  source .venv/bin/activate  (Linux/macOS)"
echo "  .venv\\Scripts\\activate    (Windows)"
echo "To update packages: ./$(basename "$0") --update"
