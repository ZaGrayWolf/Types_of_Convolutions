import torch
import torch.nn as nn
import time
import numpy as np

def calculate_depthwise_separable_flops_and_params(input_shape, output_channels, kernel_size, stride):
    in_channels = input_shape[2]
    # Compute spatial dimensions after the depthwise convolution
    out_height = (input_shape[0] - kernel_size[0]) // stride[0] + 1
    out_width = (input_shape[1] - kernel_size[1]) // stride[1] + 1

    # Depthwise convolution:
    # Parameters: kernel_height * kernel_width per channel (no bias used here)
    depthwise_params = kernel_size[0] * kernel_size[1] * in_channels
    # FLOPs for depthwise conv: multiplications, divisions, additions/subtractions per output element
    depthwise_mults = kernel_size[0] * kernel_size[1] * out_height * out_width * in_channels
    depthwise_divs = out_height * out_width * in_channels
    depthwise_adds = (kernel_size[0] * kernel_size[1] - 1) * out_height * out_width * in_channels
    depthwise_flops = depthwise_mults + depthwise_divs + depthwise_adds

    # Pointwise convolution (1x1):
    # Parameters: one weight per input channel per output channel
    pointwise_params = in_channels * output_channels
    # Each output element does: in_channels multiplications, divisions, and in_channels - 1 additions
    pointwise_mults = in_channels * out_height * out_width * output_channels
    pointwise_divs = in_channels * out_height * out_width * output_channels
    pointwise_adds = (in_channels - 1) * out_height * out_width * output_channels
    pointwise_flops = pointwise_mults + pointwise_divs + pointwise_adds

    total_params = depthwise_params + pointwise_params
    total_flops = depthwise_flops + pointwise_flops

    return total_params, total_flops, depthwise_flops, pointwise_flops, (out_height, out_width, output_channels)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def benchmark_depthwise_separable_convolution(input_shape, output_channels, kernel_size=(3, 3), stride=(2, 2)):
    in_channels = input_shape[2]
    model = DepthwiseSeparableConv(in_channels, output_channels, kernel_size, stride)
    model.eval()

    input_data = torch.randn(1, in_channels, input_shape[0], input_shape[1])

    # Warm-up run for any lazy initialization
    with torch.no_grad():
        model(input_data)
    
    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # milliseconds

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    total_params, total_flops, depthwise_flops, pointwise_flops, output_shape = \
        calculate_depthwise_separable_flops_and_params(input_shape, output_channels, kernel_size, stride)
    
    print(f"Kernel Size: {kernel_size}")
    print(f"Input Shape: {input_shape}")
    print(f"Output Shape: {output_shape}")
    print(f"Parameters: {total_params / 1e3:.2f}K")
    print(f"Total FLOPs: {total_flops / 1e6:.2f}M")
    print(f"  - Depthwise FLOPs: {depthwise_flops / 1e6:.2f}M")
    print(f"  - Pointwise FLOPs: {pointwise_flops / 1e6:.2f}M")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")

input_shape = (224, 224, 3) 

print("Benchmarking with output channels = 64 (Small Output):")
benchmark_depthwise_separable_convolution(input_shape, 64)

print("\nBenchmarking with output channels = 128 (Medium Output):")
benchmark_depthwise_separable_convolution(input_shape, 128)

print("\nBenchmarking with output channels = 256 (Large Output):")
benchmark_depthwise_separable_convolution(input_shape, 256)
