import torch
import torch.nn as nn
import time
import numpy as np

def calculate_flops_and_params(input_shape, output_channels, kernel_size, stride):
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size[0]) // stride[0] + 1
    out_width = (input_shape[1] - kernel_size[1]) // stride[1] + 1
    
    # Parameters: (kernel_height * kernel_width * in_channels + 1) * output_channels
    params = (kernel_size[0] * kernel_size[1] * in_channels + 1) * output_channels
    
    # Multiplications: (kernel_height * kernel_width * in_channels) * out_height * out_width * output_channels
    mults = (kernel_size[0] * kernel_size[1] * in_channels) * out_height * out_width * output_channels
    
    # Divisions: out_height * out_width * output_channels (for normalization)
    divs = out_height * out_width * output_channels
    
    # Additions and subtractions: (kernel_height * kernel_width * in_channels - 1) * out_height * out_width * output_channels
    add_subs = (kernel_size[0] * kernel_size[1] * in_channels - 1) * out_height * out_width * output_channels
    
    # Total FLOPs (Multiplications + Divisions + Additions/Subtractions)
    flops = mults + divs + add_subs
    
    return params, flops, mults, divs, add_subs, (out_height, out_width, output_channels)

def benchmark_convolution(input_shape, output_channels, kernel_size=(3, 3), stride=(2, 2)):
    model = nn.Conv2d(input_shape[2], output_channels, kernel_size, stride=stride)
    model.eval()
    
    # Generate random input data
    input_data = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
    
    # Warm-up run
    with torch.no_grad():
        model(input_data)
    
    # Benchmark
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
    
    params, flops, mults, divs, add_subs, output_shape = calculate_flops_and_params(input_shape, output_channels, kernel_size, stride)
    
    print(f"Kernel Size: {kernel_size}")
    print(f"Input Shape: {input_shape}")
    print(f"Output Shape: {output_shape}")
    print(f"Parameters: {params / 1e3:.2f}K")
    print(f"Total FLOPs: {flops / 1e6:.2f}M")
    print(f"Multiplications: {mults / 1e6:.2f}M")
    print(f"Divisions: {divs / 1e6:.2f}M")
    print(f"Additions and Subtractions: {add_subs / 1e9:.2f}B")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")

# Example usage
input_shape = (224, 224, 3) 
print("Benchmarking with output channels = 64 (Small Output):")
benchmark_convolution(input_shape, 64)   # Small Output

print("\nBenchmarking with output channels = 128 (Medium Output):")
benchmark_convolution(input_shape, 128)  # Medium Output

print("\nBenchmarking with output channels = 256 (Large Output):")
benchmark_convolution(input_shape, 256)  # Large Output
