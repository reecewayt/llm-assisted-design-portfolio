"""
AlexNet CNN Benchmarking Module

This module benchmarks inference performance of AlexNet on ImageNet.
"""

import time
import argparse
import os
import datetime
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm


class AlexNetBenchmark:
    """Class for benchmarking inference performance of AlexNet model."""

    def __init__(self, device=None):
        """Initialize the benchmark with AlexNet model."""
        self.model_name = 'alexnet'

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        print(f"Loading {self.model_name} on {self.device}...")
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Get preprocessing transform from the weights
        self.transform = AlexNet_Weights.DEFAULT.transforms()

    def load_imagenet(self, data_path='./data/imagenet', batch_size=32, num_workers=4):
        """Load ImageNet validation dataset."""
        try:
            # ImageNet requires manual download due to licensing
            dataset = torchvision.datasets.ImageNet(
                root=data_path,
                split='val',
                transform=self.transform
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            print(f"Loaded ImageNet validation dataset with {len(dataset)} images")
            return dataloader

        except Exception as e:
            raise FileNotFoundError(
                f"Error loading ImageNet: {e}\nPlease make sure ImageNet is correctly downloaded and placed in {data_path}"
            )

    def benchmark_inference(self, dataloader, num_batches=None):
        """Benchmark inference performance."""
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

    def display_results(self, results):
        """Display benchmark results in a readable format."""
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

    def plot_results(self, results, save_path=None):
        """Plot benchmark results."""
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
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description='AlexNet Benchmark Runner')

    parser.add_argument('--data-path', type=str, default='./data/imagenet',
                       help='Path to ImageNet dataset root directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--num-batches', type=int, default=50,
                       help='Number of batches to use (0 for all)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = AlexNetBenchmark(device=args.device)

    try:
        # Load dataset
        print(f"Loading ImageNet dataset from {args.data_path}...")
        dataloader = benchmark.load_imagenet(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

        # Set number of batches (None for all)
        num_batches = None if args.num_batches == 0 else args.num_batches

        # Run benchmark
        print(f"Starting benchmark of AlexNet...")
        results = benchmark.benchmark_inference(dataloader, num_batches=num_batches)

        # Display results
        benchmark.display_results(results)

        # Ensure results directory exists
        results_dir = './results/alexnet'
        os.makedirs(results_dir, exist_ok=True)

        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"alexnet_imagenet_{timestamp}.png"
        json_filename = f"alexnet_imagenet_{timestamp}.json"

        plot_path = os.path.join(results_dir, plot_filename)
        json_path = os.path.join(results_dir, json_filename)

        # Save plot
        benchmark.plot_results(results, save_path=plot_path)

        # Save results as JSON (exclude numpy arrays which aren't JSON serializable)
        json_results = {k: v for k, v in results.items() if k not in ['batch_times', 'accuracies']}
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Benchmark results saved to {results_dir}")
        print(f"- Plot: {plot_filename}")
        print(f"- Data: {json_filename}")

    except Exception as e:
        print(f"Error running benchmark: {e}")


if __name__ == "__main__":
    main()
