import torch
import torch.nn as nn
import time
import numpy as np

class OctaveConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha=0.5):
        """
        Implements an Octave Convolution Layer.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size (assumed square)
        :param stride: Convolution stride
        :param alpha: Ratio of low-frequency channels (0 ≤ alpha ≤ 1)
        """
        super(OctaveConv2D, self).__init__()
        self.alpha = alpha
        self.stride = stride

        # Compute channel splits
        self.out_channels_high = int((1 - alpha) * out_channels)
        self.out_channels_low = out_channels - self.out_channels_high
        self.in_channels_high = int((1 - alpha) * in_channels)
        self.in_channels_low = in_channels - self.in_channels_high

        # Define convolution layers
        self.conv_hh = nn.Conv2d(self.in_channels_high, self.out_channels_high, kernel_size, stride, padding=1)
        self.conv_hl = nn.Conv2d(self.in_channels_high, self.out_channels_low, kernel_size, stride, padding=1)
        self.conv_lh = nn.Conv2d(self.in_channels_low, self.out_channels_high, kernel_size, stride, padding=1)
        self.conv_ll = nn.Conv2d(self.in_channels_low, self.out_channels_low, kernel_size, stride, padding=1)

        # Pooling and Upsampling for frequency exchange
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Split input into high and low frequency
        x_high = x[:, :self.in_channels_high, :, :]
        x_low = self.downsample(x[:, self.in_channels_high:, :, :]) if self.in_channels_low > 0 else None

        # Compute high-frequency outputs
        y_high = self.conv_hh(x_high) + (self.upsample(self.conv_lh(x_low)) if x_low is not None else 0)

        # Compute low-frequency outputs
        y_low = self.conv_ll(x_low) + self.downsample(self.conv_hl(x_high)) if x_low is not None else None

        # Concatenate outputs
        if y_low is not None:
            return torch.cat([y_high, self.upsample(y_low)], dim=1)
        else:
            return y_high


def calculate_flops_and_params(input_shape, output_channels, kernel_size, stride, alpha=0.5):
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size) // stride + 1
    out_width = (input_shape[1] - kernel_size) // stride + 1

    # Compute channel splits
    out_channels_high = int((1 - alpha) * output_channels)
    out_channels_low = output_channels - out_channels_high
    in_channels_high = int((1 - alpha) * in_channels)
    in_channels_low = in_channels - in_channels_high

    # Compute parameters for each convolution
    params_hh = (kernel_size * kernel_size * in_channels_high + 1) * out_channels_high
    params_hl = (kernel_size * kernel_size * in_channels_high + 1) * out_channels_low
    params_lh = (kernel_size * kernel_size * in_channels_low + 1) * out_channels_high
    params_ll = (kernel_size * kernel_size * in_channels_low + 1) * out_channels_low
    total_params = params_hh + params_hl + params_lh + params_ll

    # Compute FLOPs
    mults_hh = (kernel_size * kernel_size * in_channels_high) * out_height * out_width * out_channels_high
    mults_hl = (kernel_size * kernel_size * in_channels_high) * out_height * out_width * out_channels_low
    mults_lh = (kernel_size * kernel_size * in_channels_low) * out_height * out_width * out_channels_high
    mults_ll = (kernel_size * kernel_size * in_channels_low) * out_height * out_width * out_channels_low
    total_mults = mults_hh + mults_hl + mults_lh + mults_ll

    divs = out_height * out_width * output_channels
    add_subs = (kernel_size * kernel_size * in_channels - 1) * out_height * out_width * output_channels

    total_flops = total_mults + divs + add_subs

    return total_params, total_flops, total_mults, divs, add_subs, (out_height, out_width, output_channels)


def benchmark_convolution(input_shape, output_channels, kernel_size=3, stride=2, alpha=0.5):
    model = OctaveConv2D(input_shape[2], output_channels, kernel_size, stride, alpha)
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
        input_shape, output_channels, kernel_size, stride, alpha
    )

    print(f"Octave Convolution (α={alpha})")
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
