

```python
# Basic usage with defaults (ResNet-50 on CIFAR-10)
python benchmark_runner.py

# Specify different model and dataset
python benchmark_runner.py --model resnet18 --dataset cifar100 --batch-size 64

# Benchmark on GPU (if available)
python benchmark_runner.py --device cuda
```
