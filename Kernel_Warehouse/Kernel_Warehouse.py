import torch
import torch.nn as nn
import time
import numpy as np


class KWConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_kernels=4):
        super(KWConv2D, self).__init__()
        self.kernel_size = kernel_size
        # Ensure stride is always a tuple
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.num_kernels = num_kernels
        self.out_channels = out_channels

        # Kernel bank: stores multiple kernels
        # Shape: [num_kernels, out_channels, in_channels, kernel_size, kernel_size]
        self.kernel_bank = nn.Parameter(torch.randn(num_kernels, out_channels, in_channels, kernel_size, kernel_size))

        # Attention mechanism for kernel selection
        self.attention_fc = nn.Linear(in_channels, num_kernels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channels, height, width = x.size()

        # Global Average Pooling to get channel-wise statistics
        pooled = torch.mean(x, dim=[2, 3])  # Shape: [batch, in_channels]
        attention_weights = self.attention_fc(pooled)  # Shape: [batch, num_kernels]
        attention_weights = self.softmax(attention_weights).view(batch, self.num_kernels, 1, 1, 1, 1)

        # Multiply kernel bank (unsqueezed) with attention weights and sum over kernels dimension
        # Resulting shape: [batch, out_channels, in_channels, kernel_size, kernel_size]
        selected_kernel = (self.kernel_bank.unsqueeze(0) * attention_weights).sum(dim=1)

        # Since batch size is 1 in our benchmark, we remove the extra batch dimension
        if batch == 1:
            selected_kernel = selected_kernel.squeeze(0)

        # Ensure stride is a tuple of two integers (e.g., (2, 2))
        stride_val = (int(self.stride[0]), int(self.stride[1])) if isinstance(self.stride, (tuple, list)) else (int(self.stride), int(self.stride))
        #print(f"[DEBUG] Applying convolution with stride: {stride_val}")

        # Apply convolution using the dynamically selected kernel
        conv_out = torch.nn.functional.conv2d(
            x, selected_kernel, stride=stride_val, padding=self.kernel_size // 2
        )
        return conv_out


def benchmark_convolution(input_shape, output_channels, kernel_size=3, stride=2, num_kernels=4):
    model = KWConv2D(input_shape[2], output_channels, kernel_size, stride, num_kernels)
    model.eval()

    # Generate random input data
    input_data = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])

    # Warm-up run
    with torch.no_grad():
        model(input_data)

    # Benchmark the model over 100 runs
    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert seconds to milliseconds

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Calculate some dummy FLOP/parameter numbers for demonstration
    # (In practice, dynamic convs require more careful FLOP counting.)
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size) // stride + 1
    out_width = (input_shape[1] - kernel_size) // stride + 1
    params = (kernel_size * kernel_size * in_channels + 1) * output_channels * num_kernels
    mults = (kernel_size * kernel_size * in_channels) * out_height * out_width * output_channels
    divs = out_height * out_width * output_channels
    add_subs = (kernel_size * kernel_size * in_channels - 1) * out_height * out_width * output_channels
    flops = mults + divs + add_subs
    output_shape = (out_height, out_width, output_channels)

    print("\nBenchmarking Results:")
    print(f"  Output Channels: {output_channels}")
    print(f"  Kernel Size: {kernel_size}")
    print(f"  Stride: {stride}")
    print(f"  Output Shape: {output_shape}")
    print(f"  Parameters: {params / 1e3:.2f}K")
    print(f"  Total FLOPs: {flops / 1e6:.2f}M")
    print(f"  Multiplications: {mults / 1e6:.2f}M")
    print(f"  Divisions: {divs / 1e6:.2f}M")
    print(f"  Additions/Subtractions: {add_subs / 1e6:.2f}M")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms | Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms | P99: {p99_latency:.2f}ms")


# Example usage
input_shape = (224, 224, 3)

print("\nBenchmarking with output channels = 64 (Small Output):")
benchmark_convolution(input_shape, 64)

print("\nBenchmarking with output channels = 128 (Medium Output):")
benchmark_convolution(input_shape, 128)

print("\nBenchmarking with output channels = 256 (Large Output):")
benchmark_convolution(input_shape, 256)
