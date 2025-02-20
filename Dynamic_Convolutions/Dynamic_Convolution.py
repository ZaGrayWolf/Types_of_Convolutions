import torch
import torch.nn as nn
import time
import numpy as np

class DynamicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DynamicConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Learnable parameters for dynamic weights
        self.kernel_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        batch_size, _, h, w = x.shape
        kernel_weights = self.kernel_weights  # [out_channels, in_channels, kernel_size, kernel_size]

        # Apply convolution
        output = torch.nn.functional.conv2d(x, kernel_weights, bias=self.bias, stride=self.stride, padding=1)
        return output


def calculate_flops_and_params(input_shape, output_channels, kernel_size, stride):
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size) // stride + 1
    out_width = (input_shape[1] - kernel_size) // stride + 1

    # Parameters: (kernel_height * kernel_width * in_channels + 1) * output_channels
    params = (kernel_size * kernel_size * in_channels + 1) * output_channels

    # Multiplications: (kernel_height * kernel_width * in_channels) * out_height * out_width * output_channels
    mults = (kernel_size * kernel_size * in_channels) * out_height * out_width * output_channels

    # Divisions: out_height * out_width * output_channels (for normalization)
    divs = out_height * out_width * output_channels

    # Additions and subtractions: (kernel_height * kernel_width * in_channels - 1) * out_height * out_width * output_channels
    add_subs = (kernel_size * kernel_size * in_channels - 1) * out_height * out_width * output_channels

    # Total FLOPs (Multiplications + Divisions + Additions/Subtractions)
    flops = mults + divs + add_subs

    return params, flops, mults, divs, add_subs, (out_height, out_width, output_channels)


def benchmark_convolution(input_shape, output_channels, kernel_size=3, stride=2):
    model = DynamicConv2D(input_shape[2], output_channels, kernel_size, stride)
    model.eval()

    # Generate random input data
    input_data = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])

    # Warm-up run
    with torch.no_grad():
        model(input_data)

    # Benchmarking
    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    params, flops, mults, divs, add_subs, output_shape = calculate_flops_and_params(
        input_shape, output_channels, kernel_size, stride
    )

    print(f"Dynamic Convolution")
    print(f"Kernel Size: {kernel_size}x{kernel_size}")
    print(f"Input Shape: {input_shape}")
    print(f"Output Shape: (111, 111, {output_channels})")
    print(f"Parameters: {params / 1e3:.2f}K")
    print(f"Total FLOPs: {flops / 1e6:.2f}M")
    print(f"Multiplications: {mults / 1e6:.2f}M")
    print(f"Divisions: {divs / 1e6:.2f}M")
    print(f"Additions and Subtractions: {add_subs / 1e6:.2f}M")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")
    print("\n")


# Example Usage
input_shape = (224, 224, 3)
print("Benchmarking with output channels = 64 (Small Output):")
benchmark_convolution(input_shape, 64)  # Small Output

print("\nBenchmarking with output channels = 128 (Medium Output):")
benchmark_convolution(input_shape, 128)  # Medium Output

print("\nBenchmarking with output channels = 256 (Large Output):")
benchmark_convolution(input_shape, 256)  # Large Output
