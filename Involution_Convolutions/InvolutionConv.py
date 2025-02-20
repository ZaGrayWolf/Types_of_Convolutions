import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class InvolutionConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super(InvolutionConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        
        # Kernel generation branch:
        # Generates a kernel per spatial location with shape (B, groups * kernel_size^2, H_out, W_out)
        self.kernel_gen = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups
        )
        
        # Pointwise projection to mix channels
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate dynamic kernels: shape (B, groups * kernel_size^2, H_out, W_out)
        kernel = self.kernel_gen(x)
        B, _, H_out, W_out = kernel.shape
        # Reshape kernel to (B, groups, kernel_size^2, H_out, W_out)
        kernel = kernel.view(B, self.groups, self.kernel_size * self.kernel_size, H_out, W_out)
        
        # Extract local patches from x using unfold.
        # Use same padding (kernel_size//2) as kernel_gen to ensure matching H_out, W_out.
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size // 2)
        # x_unfold has shape (B, C * kernel_size^2, H_out*W_out)
        # Reshape x_unfold to (B, groups, (C//groups)*kernel_size^2, H_out, W_out)
        x_unfold = x_unfold.view(B, self.groups, C // self.groups, self.kernel_size * self.kernel_size, H_out, W_out)
        
        # Multiply the unfolded patches with the generated kernel.
        # Expand kernel to match the channel dimension: (B, groups, 1, kernel_size^2, H_out, W_out)
        out = (x_unfold * kernel.unsqueeze(2)).sum(dim=3)  # Sum over the kernel dimension.
        # Now out has shape (B, groups, C//groups, H_out, W_out)
        
        # Reshape to (B, C, H_out, W_out)
        out = out.view(B, C, H_out, W_out)
        
        # Apply the pointwise projection
        out = self.pointwise_conv(out)
        return out

def calculate_flops_and_params(input_shape, out_channels, kernel_size, stride, in_channels):
    # Rough estimate of parameters for the two branches.
    kernel_gen_params = (in_channels * (kernel_size * kernel_size) + 1) * (kernel_size * kernel_size)
    pointwise_params = (in_channels * out_channels + 1)
    total_params = kernel_gen_params + pointwise_params

    H_out = (input_shape[0] + 2*(kernel_size//2) - kernel_size) // stride + 1
    W_out = (input_shape[1] + 2*(kernel_size//2) - kernel_size) // stride + 1

    mults = (kernel_size * kernel_size * in_channels) * H_out * W_out * out_channels
    divs = H_out * W_out * out_channels
    adds = (kernel_size * kernel_size * in_channels - 1) * H_out * W_out * out_channels
    flops = mults + divs + adds
    output_shape = (H_out, W_out, out_channels)
    return total_params, flops, mults, divs, adds, output_shape

def benchmark_convolution(input_shape, out_channels, kernel_size=3, stride=2):
    in_channels = input_shape[2]
    model = InvolutionConv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    model.eval()

    input_data = torch.randn(1, in_channels, input_shape[0], input_shape[1])
    
    # Warm-up
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
    std_latency  = np.std(latencies)
    min_latency  = np.min(latencies)
    max_latency  = np.max(latencies)
    p95_latency  = np.percentile(latencies, 95)
    p99_latency  = np.percentile(latencies, 99)

    total_params, flops, mults, divs, adds, output_shape = calculate_flops_and_params(
        input_shape, out_channels, kernel_size, stride, in_channels
    )
    
    print("\nBenchmarking Results for Involution Convolution:")
    print(f"  Output Channels: {out_channels}")
    print(f"  Kernel Size: {kernel_size}")
    print(f"  Stride: {stride}")
    print(f"  Output Shape: {output_shape}")
    print(f"  Parameters: {total_params / 1e3:.2f}K")
    print(f"  Total FLOPs: {flops / 1e6:.2f}M")
    print(f"  Multiplications: {mults / 1e6:.2f}M")
    print(f"  Divisions: {divs / 1e6:.2f}M")
    print(f"  Additions/Subtractions: {adds / 1e9:.2f}B")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms | Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms | P99: {p99_latency:.2f}ms")

if __name__ == "__main__":
    input_shape = (224, 224, 3)
    print("\nBenchmarking with output channels = 64 (Small Output):")
    benchmark_convolution(input_shape, 64, kernel_size=3, stride=2)
    print("\nBenchmarking with output channels = 128 (Medium Output):")
    benchmark_convolution(input_shape, 128, kernel_size=3, stride=2)
    print("\nBenchmarking with output channels = 256 (Large Output):")
    benchmark_convolution(input_shape, 256, kernel_size=3, stride=2)
