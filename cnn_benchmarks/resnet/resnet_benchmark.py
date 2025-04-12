"""
ResNet CNN Benchmarking Module

This module provides functionality to:
1. Load a pre-trained ResNet model
2. Import a standard dataset (CIFAR-10 or ImageNet)
3. Benchmark inference performance on the dataset
"""

import time
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class ResNetBenchmark:
    """Class for benchmarking inference performance of ResNet models."""

    AVAILABLE_MODELS = {
        'resnet18': (resnet18, ResNet18_Weights.DEFAULT),
        'resnet34': (resnet34, ResNet34_Weights.DEFAULT),
        'resnet50': (resnet50, ResNet50_Weights.DEFAULT),
        'resnet101': (resnet101, ResNet101_Weights.DEFAULT),
        'resnet152': (resnet152, ResNet152_Weights.DEFAULT)
    }

    AVAILABLE_DATASETS = ['cifar10', 'cifar100', 'imagenet']

    def __init__(self, model_name: str = 'resnet50', device: str = None):
        """
        Initialize the benchmark with specified model.

        Args:
            model_name: Name of the ResNet model to use
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name.lower()

        # Validate model name
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model using the updated weights parameter instead of pretrained
        model_fn, weights = self.AVAILABLE_MODELS[self.model_name]
        print(f"Loading {self.model_name} on {self.device}...")
        self.model = model_fn(weights=weights)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Get preprocessing transform from the weights
        self.transform = weights.transforms()

    def load_standard_dataset(self, dataset_name: str = 'cifar10', batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """
        Load a standard dataset.

        Args:
            dataset_name: Name of the dataset ('cifar10', 'cifar100', or 'imagenet')
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for data loading

        Returns:
            DataLoader for the dataset
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not available. Choose from: {self.AVAILABLE_DATASETS}")

        try:
            # Define transformations based on dataset
            if dataset_name in ['cifar10', 'cifar100']:
                # For CIFAR, we need to resize to match ResNet's expected input
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:  # ImageNet
                transform = self.transform

            # Load the appropriate dataset
            if dataset_name == 'cifar10':
                dataset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform
                )
            elif dataset_name == 'cifar100':
                dataset = torchvision.datasets.CIFAR100(
                    root='./data', train=False, download=True, transform=transform
                )
            else:  # ImageNet - This requires manual download due to licensing
                try:
                    dataset = torchvision.datasets.ImageNet(
                        root='./data/imagenet', split='val', transform=transform
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    raise FileNotFoundError(
                        "ImageNet dataset not found. Due to licensing, it must be downloaded manually. "
                        "Please download ImageNet and place it in ./data/imagenet or use CIFAR-10/100 instead."
                    ) from e

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            print(f"Loaded {dataset_name} dataset with {len(dataset)} images")
            return dataloader

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def load_custom_dataset(self, data_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """
        Load a custom dataset from the given path.

        Args:
            data_path: Path to the dataset directory
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for data loading

        Returns:
            DataLoader for the dataset
        """
        try:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(root=data_path, transform=self.transform)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            print(f"Loaded custom dataset with {len(dataset)} images from {data_path}")
            return dataloader

        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            raise

    def benchmark_inference(self, dataloader: DataLoader, num_batches: int = None) -> Dict[str, Any]:
        """
        Benchmark inference performance.

        Args:
            dataloader: DataLoader with the validation dataset
            num_batches: Number of batches to use (None for all)

        Returns:
            Dictionary with benchmark results
        """
        total_time = 0
        processed_images = 0
        batch_times = []
        accuracies = []

        # Use tqdm for progress tracking
        with torch.no_grad():  # Disable gradient calculation for inference
            for i, (images, labels) in enumerate(tqdm(dataloader, desc="Benchmarking")):
                if num_batches is not None and i >= num_batches:
                    break

                # Transfer to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.time()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / images.size(0)
                accuracies.append(accuracy)

                # Update stats
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                total_time += batch_time
                processed_images += images.size(0)

        # Compile results
        results = {
            'model': self.model_name,
            'device': self.device.type,
            'total_images': processed_images,
            'total_time': total_time,
            'images_per_second': processed_images / total_time,
            'avg_batch_time': np.mean(batch_times),
            'std_batch_time': np.std(batch_times),
            'min_batch_time': np.min(batch_times),
            'max_batch_time': np.max(batch_times),
            'avg_accuracy': np.mean(accuracies),
            'batch_times': batch_times,
            'accuracies': accuracies
        }

        return results

    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display benchmark results in a readable format.

        Args:
            results: Dictionary with benchmark results
        """
        print("\n" + "="*50)
        print(f"BENCHMARK RESULTS: {results['model']} on {results['device']}")
        print("="*50)
        print(f"Total images processed: {results['total_images']}")
        print(f"Total inference time: {results['total_time']:.2f} seconds")
        print(f"Throughput: {results['images_per_second']:.2f} images/second")
        print(f"Average batch inference time: {results['avg_batch_time']*1000:.2f} ms")
        print(f"Standard deviation of batch time: {results['std_batch_time']*1000:.2f} ms")
        print(f"Min batch time: {results['min_batch_time']*1000:.2f} ms")
        print(f"Max batch time: {results['max_batch_time']*1000:.2f} ms")
        print(f"Average accuracy: {results['avg_accuracy']*100:.2f}%")
        print("="*50)

    def plot_results(self, results: Dict[str, Any], save_path: str = None) -> None:
        """
        Plot benchmark results.

        Args:
            results: Dictionary with benchmark results
            save_path: Path to save the plots (None to display only)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot batch times
        batch_nums = list(range(1, len(results['batch_times']) + 1))
        ax1.plot(batch_nums, [t*1000 for t in results['batch_times']])
        ax1.set_title('Batch Inference Times')
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Time (ms)')
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(batch_nums, [a*100 for a in results['accuracies']])
        ax2.set_title('Batch Accuracies')
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plots saved to {save_path}")
        else:
            plt.show()


def run_benchmark_from_args():
    """Run benchmark from command line arguments."""
    parser = argparse.ArgumentParser(description='ResNet Inference Benchmark')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(ResNetBenchmark.AVAILABLE_MODELS.keys()),
                        help='ResNet model to benchmark')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=ResNetBenchmark.AVAILABLE_DATASETS + ['custom'],
                        help='Dataset to use for benchmarking')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to custom dataset directory (only used with --dataset=custom)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num-batches', type=int, default=50,
                        help='Number of batches to use (0 for all)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save the plots')

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = ResNetBenchmark(model_name=args.model, device=args.device)

    # Load dataset
    if args.dataset == 'custom' and args.data_path:
        dataloader = benchmark.load_custom_dataset(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
    else:
        dataloader = benchmark.load_standard_dataset(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

    # Set number of batches (None for all)
    num_batches = None if args.num_batches == 0 else args.num_batches

    # Run benchmark
    results = benchmark.benchmark_inference(dataloader, num_batches=num_batches)

    # Display results
    benchmark.display_results(results)

    # Plot results
    benchmark.plot_results(results, save_path=args.save_plot)

    return results


if __name__ == "__main__":
    run_benchmark_from_args()
