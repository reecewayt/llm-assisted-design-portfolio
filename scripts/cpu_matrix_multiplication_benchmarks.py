#!/usr/bin/env python3
"""
CPU Matrix Multiplication Benchmark Script

Benchmarks 3x3 matrix multiplication performance on CPU to compare against
hardware implementation that achieves 648 MOPS (Mega Operations Per Second).

The script tests multiple implementations:
1. Pure Python (nested loops)
2. NumPy with different data types
3. Manual loop unrolling
4. Pre-allocated memory approaches

Focuses on computational operations per second, minimizing data transfer overhead.
"""

import time
import numpy as np
import statistics
import platform
import psutil
import sys
from typing import List, Tuple, Callable
import gc


class MatrixBenchmark:
    def __init__(self):
        self.matrix_size = 3
        self.operations_per_matmul = 2 * (self.matrix_size ** 3)  # 54 ops for 3x3

    def print_system_info(self):
        """Print system information for context."""
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        print(f"Python Version: {sys.version}")
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print(f"CPU Count: {psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"NumPy Version: {np.__version__}")
        print()

    def generate_test_matrices(self, count: int, dtype=np.float32) -> Tuple[List, List]:
        """
        Pre-generate test matrices to minimize memory allocation during benchmark.

        Args:
            count: Number of matrix pairs to generate
            dtype: NumPy data type for matrices

        Returns:
            Tuple of (A_matrices, B_matrices) lists
        """
        np.random.seed(42)  # For reproducible results

        A_matrices = []
        B_matrices = []

        for _ in range(count):
            A = np.random.randint(-128, 127, (3, 3)).astype(dtype)
            B = np.random.randint(-128, 127, (3, 3)).astype(dtype)
            A_matrices.append(A)
            B_matrices.append(B)

        return A_matrices, B_matrices

    def pure_python_matmul(self, A: List[List], B: List[List]) -> List[List]:
        """Pure Python matrix multiplication with nested loops."""
        C = [[0 for _ in range(3)] for _ in range(3)]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i][j] += A[i][k] * B[k][j]

        return C

    def unrolled_python_matmul(self, A: List[List], B: List[List]) -> List[List]:
        """Manually unrolled Python matrix multiplication for 3x3 matrices."""
        C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # Unrolled computation for 3x3
        C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
        C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]
        C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]

        C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]
        C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]
        C[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]

        C[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0]
        C[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1]
        C[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]

        return C

    def numpy_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """NumPy matrix multiplication using @ operator."""
        return A @ B

    def numpy_dot(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """NumPy matrix multiplication using dot function."""
        return np.dot(A, B)

    def preallocated_numpy_matmul(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """NumPy matrix multiplication with pre-allocated output."""
        np.dot(A, B, out=C)
        return C

    def benchmark_function(self, func: Callable, test_data: List,
                          iterations: int = 1000000, warmup: int = 10000) -> Tuple[float, float, float, float]:
        """
        Benchmark a matrix multiplication function with separate timing for data access and computation.

        Args:
            func: Function to benchmark
            test_data: Pre-generated test data
            iterations: Number of iterations for timing
            warmup: Number of warmup iterations

        Returns:
            Tuple of (mean_time_per_op, operations_per_second, data_transfer_time, compute_time)
        """
        if len(test_data) == 2:
            A_matrices, B_matrices = test_data
            C_matrices = None
        else:
            A_matrices, B_matrices, C_matrices = test_data

        # Warmup runs
        for i in range(warmup):
            idx = i % len(A_matrices)
            if C_matrices is not None:
                func(A_matrices[idx], B_matrices[idx], C_matrices[idx])
            else:
                func(A_matrices[idx], B_matrices[idx])

        # Force garbage collection before timing
        gc.collect()

        # Separate timing for data access vs computation
        data_access_times = []
        compute_times = []
        total_times = []
        batch_size = min(1000, iterations // 10)  # Process in batches

        for batch in range(iterations // batch_size):
            batch_data_time = 0
            batch_compute_time = 0

            for i in range(batch_size):
                idx = (batch * batch_size + i) % len(A_matrices)

                # Time data access (loading from memory)
                data_start = time.perf_counter()
                A = A_matrices[idx]
                B = B_matrices[idx]
                if C_matrices is not None:
                    C = C_matrices[idx]
                data_end = time.perf_counter()

                # Time computation only
                compute_start = time.perf_counter()
                if C_matrices is not None:
                    result = func(A, B, C)
                else:
                    result = func(A, B)
                compute_end = time.perf_counter()

                batch_data_time += (data_end - data_start)
                batch_compute_time += (compute_end - compute_start)

            data_access_times.append(batch_data_time / batch_size)
            compute_times.append(batch_compute_time / batch_size)
            total_times.append((batch_data_time + batch_compute_time) / batch_size)

        mean_total_time = statistics.mean(total_times)
        mean_data_time = statistics.mean(data_access_times)
        mean_compute_time = statistics.mean(compute_times)
        ops_per_second = 1.0 / mean_total_time

        return mean_total_time, ops_per_second, mean_data_time, mean_compute_time

    def run_benchmark(self):
        """Run comprehensive benchmark of different matrix multiplication approaches."""
        print("=" * 60)
        print("CPU MATRIX MULTIPLICATION BENCHMARK (3x3)")
        print("=" * 60)
        print(f"Target to Beat: 648 MOPS (Mega Operations Per Second)")
        print(f"Operations per 3x3 matrix multiplication: {self.operations_per_matmul}")
        print()

        # Number of test matrices to pre-generate
        num_test_matrices = 1000
        iterations = 100000
        warmup = 1000

        results = []

        # Test 1: Pure Python with nested loops
        print("1. Pure Python (nested loops)...")
        A_python = []
        B_python = []
        np.random.seed(42)
        for _ in range(num_test_matrices):
            A = [[int(x) for x in row] for row in np.random.randint(-128, 127, (3, 3))]
            B = [[int(x) for x in row] for row in np.random.randint(-128, 127, (3, 3))]
            A_python.append(A)
            B_python.append(B)

        total_time, ops_per_sec, data_time, compute_time = self.benchmark_function(
            self.pure_python_matmul, (A_python, B_python), iterations, warmup
        )
        mops_total = (ops_per_sec * self.operations_per_matmul) / 1e6
        mops_compute = (self.operations_per_matmul) / (compute_time * 1e6)
        results.append(("Pure Python", ops_per_sec, mops_total, mops_compute))
        print(f"   Total time per matrix multiply: {total_time*1e6:.2f} μs")
        print(f"   Data access time: {data_time*1e6:.2f} μs ({data_time/total_time*100:.1f}%)")
        print(f"   Pure compute time: {compute_time*1e6:.2f} μs ({compute_time/total_time*100:.1f}%)")
        print(f"   Matrix multiplies per second: {ops_per_sec:.0f}")
        print(f"   Total MOPS: {mops_total:.1f}")
        print(f"   Compute-only MOPS: {mops_compute:.1f}")
        print()

        # Test 2: Unrolled Python
        print("2. Unrolled Python...")
        total_time, ops_per_sec, data_time, compute_time = self.benchmark_function(
            self.unrolled_python_matmul, (A_python, B_python), iterations, warmup
        )
        mops_total = (ops_per_sec * self.operations_per_matmul) / 1e6
        mops_compute = (self.operations_per_matmul) / (compute_time * 1e6)
        results.append(("Unrolled Python", ops_per_sec, mops_total, mops_compute))
        print(f"   Total time per matrix multiply: {total_time*1e6:.2f} μs")
        print(f"   Data access time: {data_time*1e6:.2f} μs ({data_time/total_time*100:.1f}%)")
        print(f"   Pure compute time: {compute_time*1e6:.2f} μs ({compute_time/total_time*100:.1f}%)")
        print(f"   Matrix multiplies per second: {ops_per_sec:.0f}")
        print(f"   Total MOPS: {mops_total:.1f}")
        print(f"   Compute-only MOPS: {mops_compute:.1f}")
        print()

        # Test 3: NumPy float32
        print("3. NumPy @ operator (float32)...")
        A_np32, B_np32 = self.generate_test_matrices(num_test_matrices, np.float32)
        total_time, ops_per_sec, data_time, compute_time = self.benchmark_function(
            self.numpy_matmul, (A_np32, B_np32), iterations, warmup
        )
        mops_total = (ops_per_sec * self.operations_per_matmul) / 1e6
        mops_compute = (self.operations_per_matmul) / (compute_time * 1e6)
        results.append(("NumPy @ (float32)", ops_per_sec, mops_total, mops_compute))
        print(f"   Total time per matrix multiply: {total_time*1e6:.2f} μs")
        print(f"   Data access time: {data_time*1e6:.2f} μs ({data_time/total_time*100:.1f}%)")
        print(f"   Pure compute time: {compute_time*1e6:.2f} μs ({compute_time/total_time*100:.1f}%)")
        print(f"   Matrix multiplies per second: {ops_per_sec:.0f}")
        print(f"   Total MOPS: {mops_total:.1f}")
        print(f"   Compute-only MOPS: {mops_compute:.1f}")
        print()

        # Test 4: NumPy int8 (closest to hardware)
        print("4. NumPy @ operator (int8)...")
        A_np8, B_np8 = self.generate_test_matrices(num_test_matrices, np.int8)
        total_time, ops_per_sec, data_time, compute_time = self.benchmark_function(
            self.numpy_matmul, (A_np8, B_np8), iterations, warmup
        )
        mops_total = (ops_per_sec * self.operations_per_matmul) / 1e6
        mops_compute = (self.operations_per_matmul) / (compute_time * 1e6)
        results.append(("NumPy @ (int8)", ops_per_sec, mops_total, mops_compute))
        print(f"   Total time per matrix multiply: {total_time*1e6:.2f} μs")
        print(f"   Data access time: {data_time*1e6:.2f} μs ({data_time/total_time*100:.1f}%)")
        print(f"   Pure compute time: {compute_time*1e6:.2f} μs ({compute_time/total_time*100:.1f}%)")
        print(f"   Matrix multiplies per second: {ops_per_sec:.0f}")
        print(f"   Total MOPS: {mops_total:.1f}")
        print(f"   Compute-only MOPS: {mops_compute:.1f}")
        print()

        # Test 5: NumPy dot function
        print("5. NumPy dot function (float32)...")
        total_time, ops_per_sec, data_time, compute_time = self.benchmark_function(
            self.numpy_dot, (A_np32, B_np32), iterations, warmup
        )
        mops_total = (ops_per_sec * self.operations_per_matmul) / 1e6
        mops_compute = (self.operations_per_matmul) / (compute_time * 1e6)
        results.append(("NumPy dot (float32)", ops_per_sec, mops_total, mops_compute))
        print(f"   Total time per matrix multiply: {total_time*1e6:.2f} μs")
        print(f"   Data access time: {data_time*1e6:.2f} μs ({data_time/total_time*100:.1f}%)")
        print(f"   Pure compute time: {compute_time*1e6:.2f} μs ({compute_time/total_time*100:.1f}%)")
        print(f"   Matrix multiplies per second: {ops_per_sec:.0f}")
        print(f"   Total MOPS: {mops_total:.1f}")
        print(f"   Compute-only MOPS: {mops_compute:.1f}")
        print()

        # Test 6: Pre-allocated NumPy
        print("6. NumPy with pre-allocated output (float32)...")
        C_np32 = [np.zeros((3, 3), dtype=np.float32) for _ in range(num_test_matrices)]
        total_time, ops_per_sec, data_time, compute_time = self.benchmark_function(
            self.preallocated_numpy_matmul, (A_np32, B_np32, C_np32), iterations, warmup
        )
        mops_total = (ops_per_sec * self.operations_per_matmul) / 1e6
        mops_compute = (self.operations_per_matmul) / (compute_time * 1e6)
        results.append(("NumPy pre-allocated", ops_per_sec, mops_total, mops_compute))
        print(f"   Total time per matrix multiply: {total_time*1e6:.2f} μs")
        print(f"   Data access time: {data_time*1e6:.2f} μs ({data_time/total_time*100:.1f}%)")
        print(f"   Pure compute time: {compute_time*1e6:.2f} μs ({compute_time/total_time*100:.1f}%)")
        print(f"   Matrix multiplies per second: {ops_per_sec:.0f}")
        print(f"   Total MOPS: {mops_total:.1f}")
        print(f"   Compute-only MOPS: {mops_compute:.1f}")
        print()

        # Summary
        print("=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Method':<25} {'Total MOPS':<12} {'Compute MOPS':<14} {'vs Hardware'}")
        print("-" * 75)

        hardware_mops = 648
        for name, matmul_per_sec, mops_total, mops_compute in results:
            ratio_total = mops_total / hardware_mops
            ratio_compute = mops_compute / hardware_mops
            status_total = "✓ SLOWER" if ratio_total < 1.0 else "✗ FASTER"
            status_compute = "✓ SLOWER" if ratio_compute < 1.0 else "✗ FASTER"
            print(f"{name:<25} {mops_total:<12.1f} {mops_compute:<14.1f} {ratio_compute:.3f}x {status_compute}")

        print()
        print(f"Hardware Target: {hardware_mops} MOPS")
        best_compute_mops = max(mops_compute for _, _, _, mops_compute in results)
        best_method = next(name for name, _, _, mops_compute in results if mops_compute == best_compute_mops)

        if best_compute_mops < hardware_mops:
            speedup = hardware_mops / best_compute_mops
            print(f"🎉 SUCCESS: Hardware is {speedup:.1f}x faster than best CPU compute ({best_method})")
        else:
            print(f"⚠️  WARNING: Best CPU compute method ({best_method}) achieved {best_compute_mops:.1f} MOPS")

        return results


def main():
    """Main function to run the benchmark."""
    benchmark = MatrixBenchmark()
    benchmark.print_system_info()

    # Disable NumPy multithreading to ensure pure CPU comparison
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    print("Note: NumPy multithreading disabled for fair CPU comparison")
    print()

    results = benchmark.run_benchmark()

    return results


if __name__ == "__main__":
    results = main()
