import torch
import torch.nn as nn
import time
import numpy as np

class ODConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction_ratio=4, groups=1):
        super(ODConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.out_channels = out_channels

        # Standard convolution for feature extraction
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=groups)

        # Dynamic weight generation network
        reduction_dim = max(in_channels // reduction_ratio, 4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc1 = nn.Linear(in_channels, reduction_dim)
        self.fc2 = nn.Linear(reduction_dim, out_channels)  # Generates one weight per output channel

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, height, width = x.size()

        # Generate dynamic kernel weights
        pooled = self.global_avg_pool(x).view(batch, channels)  # Global pooling
        weights = self.fc1(pooled)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights).view(batch, self.out_channels, 1, 1)  # Shape: [B, C_out, 1, 1]

        # Apply dynamic convolution
        conv_out = self.conv(x) * weights  # Channel-wise scaling
        return conv_out


def calculate_flops_and_params(input_shape, output_channels, kernel_size, stride, groups=1):
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size) // stride + 1
    out_width = (input_shape[1] - kernel_size) // stride + 1

    # Parameters: Convolution + Dynamic Weight Network
    conv_params = (kernel_size * kernel_size * in_channels // groups + 1) * output_channels
    fc1_params = (in_channels * (in_channels // 4))
    fc2_params = ((in_channels // 4) * output_channels)  # Fixed here
    total_params = conv_params + fc1_params + fc2_params

    # FLOP Calculation
    conv_mults = (kernel_size * kernel_size * in_channels // groups) * out_height * out_width * output_channels
    fc1_mults = in_channels * (in_channels // 4)
    fc2_mults = (in_channels // 4) * output_channels  # Fixed here
    total_mults = conv_mults + fc1_mults + fc2_mults

    divs = out_height * out_width * output_channels
    add_subs = (kernel_size * kernel_size * in_channels - 1) * out_height * out_width * output_channels

    total_flops = total_mults + divs + add_subs

    return total_params, total_flops, total_mults, divs, add_subs, (out_height, out_width, output_channels)


def benchmark_convolution(input_shape, output_channels, kernel_size=3, stride=2, groups=1):
    model = ODConv2D(input_shape[2], output_channels, kernel_size, stride, groups=groups)
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
        input_shape, output_channels, kernel_size, stride, groups
    )

    print(f"Omni-Dimensional Dynamic Convolution (ODConv)")
    print(f"Kernel Size: {kernel_size}x{kernel_size}")
    print(f"Input Shape: {input_shape}")
    print(f"Output Shape: {output_shape}")
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
