"""
CNN Benchmark Runner

This module serves as a centralized entry point to run various CNN architecture benchmarks.
Currently supports:
- ResNet
- AlexNet
"""

import os
import argparse
import sys
import datetime
import json
from pathlib import Path


def run_resnet_benchmark(args):
    """Run ResNet benchmark with given arguments."""
    try:
        # Add the resnet directory to the Python path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'resnet'))
        from resnet.resnet_benchmark import ResNetBenchmark

        # Setup directories for results
        results_dir = os.path.join(os.path.dirname(__file__), 'results', 'resnet')
        os.makedirs(results_dir, exist_ok=True)

        # Create benchmark instance
        benchmark = ResNetBenchmark(model_name=args.model, device=args.device)

        # Load dataset
        print(f"Loading {args.dataset} dataset...")
        dataloader = benchmark.load_standard_dataset(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

        # Set number of batches (None for all)
        num_batches = None if args.num_batches == 0 else args.num_batches

        # Run benchmark
        print(f"Starting benchmark of {args.model}...")
        results = benchmark.benchmark_inference(dataloader, num_batches=num_batches)

        # Display results
        benchmark.display_results(results)

        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{args.model}_{args.dataset}_{timestamp}.png"
        json_filename = f"{args.model}_{args.dataset}_{timestamp}.json"

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

        return results

    except ImportError as e:
        print(f"Error importing ResNet benchmark module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running ResNet benchmark: {e}")
        sys.exit(1)


def run_alexnet_benchmark(args):
    """Run AlexNet benchmark with given arguments."""
    try:
        # Add the alexnet directory to the Python path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'alexnet'))
        from alexnet.alexnet_benchmark import AlexNetBenchmark

        # Setup directories for results
        results_dir = os.path.join(os.path.dirname(__file__), 'results', 'alexnet')
        os.makedirs(results_dir, exist_ok=True)

        # Create benchmark instance
        benchmark = AlexNetBenchmark(device=args.device)

        # Load dataset
        print(f"Loading {args.dataset} dataset...")
        dataloader = benchmark.load_standard_dataset(
            dataset_name=args.dataset,
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

        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"alexnet_{args.dataset}_{timestamp}.png"
        json_filename = f"alexnet_{args.dataset}_{timestamp}.json"

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

        return results

    except ImportError as e:
        print(f"Error importing AlexNet benchmark module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running AlexNet benchmark: {e}")
        sys.exit(1)


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description='CNN Benchmark Runner')

    # Main arguments
    parser.add_argument('--architecture', type=str, default='resnet',
                        choices=['resnet', 'alexnet'],
                        help='CNN architecture to benchmark')

    # Common benchmark arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Specific model within the architecture (for ResNet only)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use for benchmarking')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num-batches', type=int, default=50,
                        help='Number of batches to use (0 for all)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Run the appropriate benchmark
    if args.architecture == 'resnet':
        run_resnet_benchmark(args)
    elif args.architecture == 'alexnet':
        run_alexnet_benchmark(args)
    else:
        print(f"Architecture {args.architecture} is not supported yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
