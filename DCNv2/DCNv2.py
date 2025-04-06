import torch
import torch.nn as nn
import time
import numpy as np
from torchvision.ops import DeformConv2d

class DeformConv2dPack(nn.Module):
    """
    Implements the Deformable Convolution v2 (DCNv2) layer.
    It uses an internal conv (offset_conv) to predict offsets and modulation masks,
    then applies deformable convolution.
    If the installed torchvision.ops.DeformConv2d does not support the mask argument,
    the forward pass falls back to using only the offset.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(DeformConv2dPack, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.deformable_groups = deformable_groups

        # For each kernel element, predict 2 offsets and 1 modulation value.
        offset_channels = deformable_groups * 2 * kernel_size[0] * kernel_size[1]
        mask_channels   = deformable_groups * 1 * kernel_size[0] * kernel_size[1]
        self.offset_mask_channels = offset_channels + mask_channels

        # Convolution to predict offsets and masks.
        self.offset_conv = nn.Conv2d(in_channels, self.offset_mask_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation)
        # Deformable convolution operator.
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation,
                                        groups=groups, bias=bias)

    def forward(self, x):
        # Predict offsets and modulation masks.
        out = self.offset_conv(x)
        offset_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        offset = out[:, :offset_channels, :, :]
        mask = out[:, offset_channels:, :, :]
        mask = torch.sigmoid(mask)
        try:
            return self.deform_conv(x, offset, mask)
        except TypeError:
            return self.deform_conv(x, offset)

def compute_flops(model, output):
    """
    Manually compute the FLOPs for the module.
    For a standard convolution:
      FLOPs = 2 * (in_channels * kernel_height * kernel_width) * (output_pixels) * (output_channels)
    If a bias is used, add one operation per output element.
    
    For deformable convolution, we add an extra cost per kernel element at each output location
    to account for the bilinear interpolation (assumed here as 7 extra operations per sample).
    """
    # Get output dimensions (from deformable conv output)
    N, C_out, H_out, W_out = output.shape
    output_pixels = H_out * W_out

    # FLOPs for offset_conv (a standard convolution)
    w_offset = model.offset_conv.weight
    out_channels_offset, in_channels_offset, kH_offset, kW_offset = w_offset.shape
    flops_offset = 2 * (in_channels_offset * kH_offset * kW_offset) * output_pixels * out_channels_offset
    if model.offset_conv.bias is not None:
        flops_offset += output_pixels * out_channels_offset

    # FLOPs for deform_conv (base convolution part)
    w_deform = model.deform_conv.weight
    out_channels_deform, in_channels_deform, kH_deform, kW_deform = w_deform.shape
    kernel_area = kH_deform * kW_deform
    # Standard convolution FLOPs for deform_conv.
    flops_deform_base = 2 * (in_channels_deform * kernel_area) * output_pixels * out_channels_deform
    if model.deform_conv.bias is not None:
        flops_deform_base += output_pixels * out_channels_deform
    # Extra FLOPs due to bilinear interpolation (approx. 7 ops per kernel element per output pixel).
    flops_deform_extra = 7 * kernel_area * output_pixels * out_channels_deform
    flops_deform = flops_deform_base + flops_deform_extra

    total_flops = flops_offset + flops_deform
    return total_flops

def benchmark_convolution(input_shape, out_channels, kernel_size=3, stride=1):
    """
    Benchmarks the deformable convolution module:
      - Creates a dummy input (expected shape: (H, W, C))
      - Runs a forward pass to obtain output shape.
      - Counts parameters.
      - Computes FLOPs using a manual computation (with extra bilinear interpolation overhead).
      - Measures latency (mean, std, min, max, P95, and P99) over multiple iterations.
    """
    H, W, C = input_shape
    in_channels = C
    padding = kernel_size // 2

    model = DeformConv2dPack(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, deformable_groups=1)
    model.eval()

    input_tensor = torch.randn(1, in_channels, H, W)
    with torch.no_grad():
        output = model(input_tensor)
    output_shape = tuple(output.shape)
    total_params = sum(p.numel() for p in model.parameters())

    total_flops = compute_flops(model, output)

    # Breakdown of FLOPs.
    mults = total_flops / 2
    adds  = total_flops / 2
    divs  = 0

    # Benchmark latency.
    iterations = 100
    latencies = []
    with torch.no_grad():
        for _ in range(10):  # warm-up runs
            _ = model(input_tensor)
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            latencies.append((end - start) * 1000)
    latencies = np.array(latencies)
    mean_latency = latencies.mean()
    std_latency = latencies.std()
    min_latency = latencies.min()
    max_latency = latencies.max()
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Print the results.
    print("\nBenchmarking Results for Deformable Convolution v2:")
    print(f"  Output Channels: {out_channels}")
    print(f"  Kernel Size: {kernel_size}")
    print(f"  Stride: {stride}")
    print(f"  Output Shape: {output_shape}")
    print(f"  Parameters: {total_params / 1e3:.2f}K")
    print(f"  Total FLOPs: {total_flops / 1e6:.2f}M")
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
