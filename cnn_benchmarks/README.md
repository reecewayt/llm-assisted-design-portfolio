# CNN Benchmarking Tool

This project provides a tool for benchmarking the inference performance of Convolutional Neural Networks (CNNs). Currently, it supports benchmarking AlexNet on the ImageNet dataset. For more details and long form context, please see the accompanying wiki page. [!TODO:]. I've chosen AlexNet because of its simple architecture, and ImageNet because the pretrained model through pytorch was trained on this dataset. Hence, this project is more concerned with inference benchmarking than training for other datasets.

## Project Structure

```
cnn_benchmarks/
├── README.md
├── alexnet/
│   └── alexnet_benchmark.py    # AlexNet benchmark implementation
├── data/
│   └── imagenet/               # ImageNet dataset
│       ├── val/                # Validation images (structured)
│       └── ...
├── results/
│   └── alexnet/                # Benchmark results
│       ├── *.json              # Metrics in JSON format
│       └── *.png               # Performance plots
└── setup_imagenet.sh           # Script to set up ImageNet
```

## Prerequisites
Before downloading imagenet and getting started with this project. Please run the `setup.sh` script in the root directory, which instantiates a python venv, and install the necessary python packages from the `requirements.txt` file.

```bash
cd path/to/llm-assisted-design-portfolio
source ./setup.sh

```

## Downloading ImageNet

To run the benchmark, you need the ImageNet validation dataset. Follow these steps to download it:

1. Create an account on [ImageNet](https://image-net.org/index.php)
2. Log in and navigate to the "Download" section
3. Request access to the ImageNet dataset (this may take some time to be approved)
4. Download the following files:
   - `ILSVRC2012_img_val.tar` (validation images)
   - `ILSVRC2012_devkit_t12.tar.gz` (development toolkit)

5. Make the directory, `data/imagenet`, and place these files in the `data/imagenet/` directory

## Setting Up ImageNet

After downloading the required files, run the provided setup script to organize the ImageNet validation dataset:

```bash
# Make the script executable
chmod +x setup_imagenet.sh

# Run the setup script
./setup_imagenet.sh
```

The script will:
1. Create the necessary directory structure
2. Extract the validation images
3. Organize them into the appropriate class folders

## Running the Benchmark

To run the AlexNet benchmark on the ImageNet validation dataset:

```bash
python alexnet/alexnet_benchmark.py --data-path ./data/imagenet
```

### Options

- `--data-path`: Path to the ImageNet dataset directory (default: `./data/imagenet`)
- `--batch-size`: Batch size for inference (default: 32)
- `--num-batches`: Number of batches to use (0 for all, default: 50)
- `--device`: Device to run on (cuda/cpu, default: auto-detect)
- `--workers`: Number of data loading workers (default: 4)

## Benchmark Results

After running the benchmark, results will be saved in the `results/alexnet/` directory:
- JSON file with metrics (throughput, accuracy, etc.)
- PNG plot of batch times and accuracies

The benchmark reports:
- Total images processed
- Total inference time
- Throughput (images/second)
- Average batch inference time
- Standard deviation of batch time
- Minimum and maximum batch times
- Average accuracy

## Example Results

```
==================================================
BENCHMARK RESULTS: alexnet on cuda
==================================================
Total images processed: 1600
Total inference time: 2.40 seconds
Throughput: 666.40 images/second
Average batch inference time: 48.02 ms
Standard deviation of batch time: 224.87 ms
Min batch time: 15.65 ms
Max batch time: 1622.09 ms
Average accuracy: 73.94%
```
