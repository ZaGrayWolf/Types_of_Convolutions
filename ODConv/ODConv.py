import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class ODConv2D(nn.Module):
    """
    Omni-Dimensional Dynamic Convolution (ODConv) layer.
    
    This layer generates dynamic convolution kernels by applying dynamic attention 
    mechanisms along four dimensions:
      - Spatial (kernel height and width)
      - Input Channel
      - Output Channel
      - Kernel Weight (element-wise modulation)
    
    The final dynamic kernel is computed as:
      Dynamic_Kernel = W ⊙ α^s ⊙ α^c ⊙ α^f ⊙ α^w
    where:
      - W is the fixed base kernel of shape [C_out, C_in, K, K].
      - α^s is spatial attention, output shape [1, 1, 1, K, K]
      - α^c is input channel attention, shape [1, 1, C_in, 1, 1]
      - α^f is output channel attention, shape [1, C_out, 1, 1, 1]
      - α^w is kernel weight attention, shape [1, C_out, C_in, K, K]
    
    All attention factors are generated from a global context vector obtained via 
    global average pooling on the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction_ratio=4):
        super(ODConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # assume square kernels
        self.stride = stride
        
        # Base (static) kernel W: shape [C_out, C_in, K, K]
        self.base_kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Global Average Pooling to get context vector (shape: [B, in_channels])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Four separate fully connected branches to generate dynamic attention:
        # 1. Spatial attention: output vector of size (K*K)
        self.fc_spatial = nn.Linear(in_channels, kernel_size * kernel_size)
        # 2. Input channel attention: output vector of size (in_channels)
        self.fc_in = nn.Linear(in_channels, in_channels)
        # 3. Output channel attention: output vector of size (out_channels)
        self.fc_out = nn.Linear(in_channels, out_channels)
        # 4. Kernel weight attention: output vector of size (C_out * C_in * K*K)
        self.fc_kernel = nn.Linear(in_channels, out_channels * in_channels * kernel_size * kernel_size)
        
        # Activation function (you could use softmax or sigmoid; here we use sigmoid for element-wise scaling)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch, channels, H, W = x.size()
        
        # Get the global context vector: shape [B, in_channels]
        context = self.global_avg_pool(x).view(batch, self.in_channels)
        
        # Generate dynamic attention factors for each dimension:
        # Spatial attention: shape [B, K*K] → reshape to [B, 1, 1, K, K]
        attn_spatial = self.sigmoid(self.fc_spatial(context)).view(batch, 1, 1, self.kernel_size, self.kernel_size)
        
        # Input channel attention: shape [B, in_channels] → reshape to [B, 1, C_in, 1, 1]
        attn_in = self.sigmoid(self.fc_in(context)).view(batch, 1, self.in_channels, 1, 1)
        
        # Output channel attention: shape [B, out_channels] → reshape to [B, C_out, 1, 1, 1]
        attn_out = self.sigmoid(self.fc_out(context)).view(batch, self.out_channels, 1, 1, 1)
        
        # Kernel weight attention: shape [B, C_out * C_in * K*K] → reshape to [B, C_out, C_in, K, K]
        attn_kernel = self.sigmoid(self.fc_kernel(context)).view(batch, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # The final dynamic kernel for each sample is computed by combining these factors with the base kernel.
        # Base kernel: shape [C_out, C_in, K, K] -> expand to [1, C_out, C_in, K, K] for broadcast.
        base = self.base_kernel.unsqueeze(0)
        # Compute dynamic kernel: multiply element-wise all factors.
        # The broadcasted shapes:
        # attn_out: [B, C_out, 1, 1, 1]
        # attn_in:  [B, 1, C_in, 1, 1]
        # attn_spatial: [B, 1, 1, K, K]
        # attn_kernel: [B, C_out, C_in, K, K]
        dynamic_kernel = base * attn_out * attn_in * attn_spatial * attn_kernel  # [B, C_out, C_in, K, K]
        
        # For batch size 1, squeeze out the batch dimension to match F.conv2d requirement.
        if batch == 1:
            dynamic_kernel = dynamic_kernel.squeeze(0)  # [C_out, C_in, K, K]
        
        # Perform convolution using the dynamic kernel. (Note: For batch >1, F.conv2d doesn't accept per-sample kernels.)
        out = F.conv2d(x, dynamic_kernel, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
        return out

# Example benchmark function remains similar to before.
def calculate_flops_and_params(input_shape, out_channels, kernel_size, stride, in_channels):
    # For simplicity, we compute parameter count as:
    # Base kernel: (C_out * C_in * K*K)
    base_params = out_channels * in_channels * (kernel_size * kernel_size)
    # Attention branch parameters:
    fc_spatial_params = in_channels * (kernel_size * kernel_size)  # fc_spatial
    fc_in_params = in_channels * in_channels
    fc_out_params = in_channels * out_channels
    fc_kernel_params = in_channels * (out_channels * in_channels * kernel_size * kernel_size)
    attn_params = fc_spatial_params + fc_in_params + fc_out_params + fc_kernel_params
    total_params = base_params + attn_params
    
    # FLOPs (dummy estimation): Use static convolution formula for base kernel.
    out_height = (input_shape[0] + 2*(kernel_size//2) - kernel_size) // stride + 1
    out_width = (input_shape[1] + 2*(kernel_size//2) - kernel_size) // stride + 1
    mults = (kernel_size * kernel_size * in_channels) * out_height * out_width * out_channels
    divs = out_height * out_width * out_channels
    adds = ((kernel_size * kernel_size * in_channels) - 1) * out_height * out_width * out_channels
    conv_flops = mults + divs + adds
    # Extra FLOPs from attention branch (rough estimation):
    gap_flops = in_channels * (input_shape[0] * input_shape[1])
    fc_flops = (in_channels * (kernel_size * kernel_size) + in_channels * in_channels + in_channels * out_channels + in_channels * (out_channels * in_channels * kernel_size * kernel_size))
    extra_flops = gap_flops + fc_flops
    total_flops = conv_flops + extra_flops
    output_shape = (out_height, out_width, out_channels)
    return total_params, total_flops, mults, divs, adds, output_shape

def benchmark_convolution(input_shape, out_channels, kernel_size=3, stride=2):
    in_channels = input_shape[2]
    model = ODConv2D(in_channels, out_channels, kernel_size, stride)
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
    
    total_params, total_flops, mults, divs, adds, output_shape = calculate_flops_and_params(
        input_shape, out_channels, kernel_size, stride, in_channels
    )
    
    print("\nBenchmarking Results for ODConv with Dynamic Attention on 4 Dimensions:")
    print(f"  Output Channels: {out_channels}")
    print(f"  Kernel Size: {kernel_size}")
    print(f"  Stride: {stride}")
    print(f"  Output Shape: {output_shape}")
    print(f"  Parameters: {total_params / 1e3:.2f}K")
    print(f"  Total FLOPs: {total_flops / 1e6:.2f}M")
    print(f"  Multiplications: {mults / 1e6:.2f}M")
    print(f"  Divisions: {divs / 1e6:.2f}M")
    print(f"  Additions/Subtractions: {adds / 1e6:.2f}M")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms | Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms | P99: {p99_latency:.2f}ms")
    print("\n")


# Example usage:
input_shape = (224, 224, 3)

print("Benchmarking with output channels = 64 (Small Output):")
benchmark_convolution(input_shape, 64)

print("\nBenchmarking with output channels = 128 (Medium Output):")
benchmark_convolution(input_shape, 128)

print("\nBenchmarking with output channels = 256 (Large Output):")
benchmark_convolution(input_shape, 256)
