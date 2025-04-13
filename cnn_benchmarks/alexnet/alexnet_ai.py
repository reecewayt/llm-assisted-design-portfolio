import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
import math

def calculate_conv_flops(in_channels, out_channels, kernel_size, input_size):
    """Calculate FLOPs for a convolutional layer"""
    # For each output element, we do:
    # kernel_size^2 * in_channels multiplications and additions
    operations_per_element = kernel_size**2 * in_channels * 2  # multiply-add
    # Output size after convolution
    output_elements = input_size**2
    return operations_per_element * output_elements * out_channels

def calculate_conv_memory(in_channels, out_channels, kernel_size, input_size, batch_size=1, bytes_per_element=4):
    """Calculate memory accesses for a convolutional layer in bytes"""
    # Memory reads: input feature map + weights
    input_elements = input_size**2 * in_channels * batch_size
    weight_elements = kernel_size**2 * in_channels * out_channels
    bias_elements = out_channels

    # Memory writes: output feature map
    output_size = input_size  # Assuming padding preserves dimensions
    output_elements = output_size**2 * out_channels * batch_size

    # Total memory in bytes
    total_memory = (input_elements + weight_elements + bias_elements + output_elements) * bytes_per_element
    return total_memory

def calculate_fc_flops(in_features, out_features):
    """Calculate FLOPs for a fully connected layer"""
    # For each output, we do in_features multiplications and additions
    return in_features * out_features * 2  # multiply-add

def calculate_fc_memory(in_features, out_features, batch_size=1, bytes_per_element=4):
    """Calculate memory accesses for a fully connected layer in bytes"""
    # Memory reads: input + weights
    input_elements = in_features * batch_size
    weight_elements = in_features * out_features
    bias_elements = out_features

    # Memory writes: output
    output_elements = out_features * batch_size

    # Total memory in bytes
    total_memory = (input_elements + weight_elements + bias_elements + output_elements) * bytes_per_element
    return total_memory

def analyze_alexnet_intensity():
    # Load the model
    alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    # Input image size for AlexNet
    input_size = 224
    batch_size = 1

    print("\n===== ARITHMETIC INTENSITY ANALYSIS =====")
    print("Layer                   | FLOPs (M)   | Memory (MB)  | Arithmetic Intensity (FLOPs/Byte)")
    print("----------------------------------------------------------------------------------")

    # Track sizes through the network
    current_size = input_size
    current_channels = 3

    # Analyze convolutional layers
    for i, layer in enumerate(alexnet.features):
        if isinstance(layer, torch.nn.Conv2d):
            # Extract layer parameters
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size[0]  # Assuming square kernel
            stride = layer.stride[0]  # Assuming square stride
            padding = layer.padding[0]  # Assuming same padding on all sides

            # Calculate output size after this layer
            output_size = math.floor((current_size + 2*padding - kernel_size) / stride + 1)

            # Calculate FLOPs and memory
            flops = calculate_conv_flops(in_channels, out_channels, kernel_size, output_size)
            memory = calculate_conv_memory(in_channels, out_channels, kernel_size, current_size)

            # Calculate arithmetic intensity
            intensity = flops / memory

            print(f"Conv2d {i:<16} | {flops/1e6:10.2f} | {memory/1e6:11.2f} | {intensity:10.2f}")

            # Update for next layer
            current_size = output_size
            current_channels = out_channels

        elif isinstance(layer, torch.nn.MaxPool2d):
            # Calculate output size after pooling
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding

            # If kernel_size is an int, convert to tuple
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)

            output_size = math.floor((current_size + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
            current_size = output_size

    # After features, there's the adaptive avgpool to 6x6
    current_size = 6  # Fixed output size of AdaptiveAvgPool2d in AlexNet

    # Analyze fully connected layers
    fc_sizes = [(current_size*current_size*current_channels, 4096), (4096, 4096), (4096, 1000)]
    for i, (in_features, out_features) in enumerate(fc_sizes):
        flops = calculate_fc_flops(in_features, out_features)
        memory = calculate_fc_memory(in_features, out_features)
        intensity = flops / memory

        print(f"Linear {i:<16} | {flops/1e6:10.2f} | {memory/1e6:11.2f} | {intensity:10.2f}")

if __name__ == "__main__":
    analyze_alexnet_intensity()
