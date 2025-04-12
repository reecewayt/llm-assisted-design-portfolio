#!/bin/bash
# Script to set up ImageNet dataset for benchmarking

# Set the base directory for the ImageNet dataset
BASE_DIR="./data/imagenet"
VAL_DIR="$BASE_DIR/val"

echo "Setting up ImageNet dataset in $BASE_DIR"

# Create directories if they don't exist
mkdir -p "$BASE_DIR"
mkdir -p "$VAL_DIR"

# Check if the required files exist
VAL_TAR="$BASE_DIR/ILSVRC2012_img_val.tar"
DEVKIT_TAR="$BASE_DIR/ILSVRC2012_devkit_t12.tar.gz"

if [ ! -f "$VAL_TAR" ]; then
  echo "Error: $VAL_TAR not found!"
  echo "Please download the validation dataset from https://image-net.org/index.php"
  echo "and place it in $BASE_DIR"
  exit 1
fi

if [ ! -f "$DEVKIT_TAR" ]; then
  echo "Warning: $DEVKIT_TAR not found!"
  echo "This file is recommended for full functionality."
  echo "You can download it from https://image-net.org/index.php"
fi

# Extract validation images if not already extracted
NUM_VAL_IMAGES=$(find "$VAL_DIR" -name "*.JPEG" | wc -l)
if [ "$NUM_VAL_IMAGES" -lt 1000 ]; then
  echo "Extracting validation images from $VAL_TAR"
  tar -xf "$VAL_TAR" -C "$VAL_DIR"
else
  echo "Validation images already extracted"
fi

# Extract development kit if available
if [ -f "$DEVKIT_TAR" ]; then
  DEVKIT_DIR="$BASE_DIR/devkit"
  mkdir -p "$DEVKIT_DIR"

  if [ ! -d "$DEVKIT_DIR/data" ]; then
    echo "Extracting development kit from $DEVKIT_TAR"
    tar -xzf "$DEVKIT_TAR" -C "$DEVKIT_DIR"
  else
    echo "Development kit already extracted"
  fi
fi

# Download the valprep.sh script to organize images into class folders
VALPREP_SCRIPT="$BASE_DIR/valprep.sh"

if [ ! -f "$VALPREP_SCRIPT" ]; then
  echo "Downloading valprep.sh script"
  wget -O "$VALPREP_SCRIPT" https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
  chmod +x "$VALPREP_SCRIPT"
fi

# Run the valprep script to organize images
echo "Organizing validation images into class folders"
cd "$VAL_DIR"
bash "../valprep.sh"

echo "Setup complete! You can now run the benchmark with:"
echo "python alexnet/alexnet_benchmark.py --data-path ./data/imagenet"
