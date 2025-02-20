import torch
import torch.nn as nn
import time
import numpy as np

def calculate_flops_and_params(input_shape, output_channels, kernel_size, stride, mode):
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size[0]) // stride[0] + 1
    out_width = (input_shape[1] - kernel_size[1]) // stride[1] + 1
    
    if mode == 'depthwise':
        params = (kernel_size[0] * kernel_size[1] * in_channels) + in_channels
        mults = (kernel_size[0] * kernel_size[1] * in_channels) * out_height * out_width
    elif mode == 'pointwise':
        params = (in_channels + 1) * output_channels
        mults = in_channels * out_height * out_width * output_channels
    else:  # Depthwise Separable
        params = (kernel_size[0] * kernel_size[1] * in_channels) + in_channels + (in_channels * output_channels) + output_channels
        mults = (kernel_size[0] * kernel_size[1] * in_channels) * out_height * out_width + in_channels * out_height * out_width * output_channels
    
    divs = out_height * out_width * output_channels
    add_subs = (kernel_size[0] * kernel_size[1] * in_channels - 1) * out_height * out_width * output_channels
    
    flops = mults + divs + add_subs
    
    return params, flops, mults, divs, add_subs, (out_height, out_width, output_channels)

def benchmark_convolution(input_shape, output_channels, kernel_size=(3, 3), stride=(2, 2), mode='depthwise'):
    in_channels = input_shape[2]
    if mode == 'depthwise':
        model = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels)
    elif mode == 'pointwise':
        model = nn.Conv2d(in_channels, output_channels, (1, 1), stride=(1, 1))
    else:  # Depthwise Separable
        model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels),
            nn.Conv2d(in_channels, output_channels, (1, 1))
        )
    model.eval()
    
    input_data = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
    with torch.no_grad():
        model(input_data)
    
    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)
    
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    params, flops, mults, divs, add_subs, output_shape = calculate_flops_and_params(
        input_shape, output_channels, kernel_size, stride, mode
    )
    
    print(f"{mode.capitalize()} Convolution")
    print(f"Kernel Size: {kernel_size}")
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

input_shape = (224, 224, 3)

print("Benchmarking with output channels = 64 (Small Output):")
benchmark_convolution(input_shape, 64, mode='depthwise')

print("\nBenchmarking with output channels = 128 (Medium Output):")
benchmark_convolution(input_shape, 128, mode='depthwise')

print("\nBenchmarking with output channels = 256 (Large Output):")
benchmark_convolution(input_shape, 256, mode='depthwise')
