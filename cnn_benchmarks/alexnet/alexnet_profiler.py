import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
from torch.profiler import profile, record_function, ProfilerActivity

def check_gpu_availability():
    print("\n===== GPU DETECTION DIAGNOSTICS =====")
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get device count and properties
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")

        for i in range(device_count):
            print(f"\nCUDA Device {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

        # Check current device
        print(f"\nCurrent CUDA device: {torch.cuda.current_device()}")

        # Test tensor creation on GPU
        try:
            test_tensor = torch.tensor([1, 2, 3], device="cuda")
            print(f"Test tensor created on GPU: {test_tensor.device}")
            del test_tensor  # Clean up
            print("GPU tensor test: PASSED")
        except Exception as e:
            print(f"GPU tensor test: FAILED - {str(e)}")

    # Check XPU availability if applicable
    if hasattr(torch, 'xpu'):
        xpu_available = torch.xpu.is_available()
        print(f"\nXPU available: {xpu_available}")

        if xpu_available:
            xpu_device_count = torch.xpu.device_count()
            print(f"Number of XPU devices: {xpu_device_count}")

    print("\nDefault device that will be used:", end=" ")
    if cuda_available:
        print("CUDA (GPU)")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("XPU")
    else:
        print("CPU")


def explore_alexnet_architecture():
    # Load the AlexNet model with the updated weights parameter
    alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    # Print the high-level model structure
    print("===== COMPLETE MODEL STRUCTURE =====")
    print(alexnet)
    print("\n")

    # Print the features part (convolutional layers)
    print("===== FEATURES SECTION (CONV LAYERS) =====")
    for idx, layer in enumerate(alexnet.features):
        print(f"Layer {idx}: {layer}")
    print("\n")

    # Print the avgpool layer
    print("===== AVGPOOL LAYER =====")
    print(alexnet.avgpool)
    print("\n")

    # Print the classifier part (fully connected layers)
    print("===== CLASSIFIER SECTION (FC LAYERS) =====")
    for idx, layer in enumerate(alexnet.classifier):
        print(f"Layer {idx}: {layer}")
    print("\n")

    # Print parameter counts for each section
    print("===== PARAMETER COUNTS =====")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {count_parameters(alexnet):,}")
    print(f"Features parameters: {count_parameters(alexnet.features):,}")
    print(f"Classifier parameters: {count_parameters(alexnet.classifier):,}")

    # Print layer names and shapes
    print("\n===== LAYER NAMES AND SHAPES =====")
    for name, param in alexnet.named_parameters():
        print(f"{name}: {param.shape}")


def profile_alexnet():
    print("\n===== PYTORCH PROFILER RESULTS =====")

    # Determine the available device and exit if no acceleration is available
    if torch.cuda.is_available():
        device = 'cuda'
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
        activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]
    else:
        device = 'cpu'
        activities = [ProfilerActivity.CPU]

    print(f"Profiling on device: {device}")

    # Set the sort keyword based on available device
    sort_by_keyword = "cpu_time_total" if device == 'cpu' else f"{device}_time_total"

    # Load model and prepare input
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    inputs = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up run
    with torch.no_grad():
        model(inputs)

    # Profile the model
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(inputs)

    # Print profiling results
    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=20))


if __name__ == "__main__":
    check_gpu_availability()
    explore_alexnet_architecture()
    profile_alexnet()
