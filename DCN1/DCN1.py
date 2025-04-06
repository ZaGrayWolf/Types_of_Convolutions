import math

def calculate_conv_properties(input_size, in_channels, out_channels, kernel_size, deformable=False):
    H, W = input_size
    K = kernel_size ** 2  # Number of kernel points
    
    # Standard convolution parameters
    std_params = (kernel_size**2 * in_channels + 1) * out_channels
    
    # Standard convolution FLOPs (MAC operations)
    std_flops = 2 * (kernel_size**2 * in_channels) * out_channels * H * W
    
    if not deformable:
        return std_params, std_flops
    
    # DCNv1 components
    # Offset prediction layer parameters (3x3 conv -> 2K channels)
    offset_params = (kernel_size**2 * in_channels + 1) * 2 * K
    
    # Offset prediction FLOPs
    offset_flops = 2 * (kernel_size**2 * in_channels) * (2 * K) * H * W
    
    # Deformable interpolation FLOPs (4 ops per sample)
    interp_flops = K * 4 * H * W * out_channels
    
    total_params = std_params + offset_params
    total_flops = std_flops + offset_flops + interp_flops
    
    return total_params, total_flops

# User-provided configurations
configs = [
    {"input_size": (224, 224), "in_channels": 3, "out_channels": 64, "kernel_size": 3},
    {"input_size": (224, 224), "in_channels": 3, "out_channels": 128, "kernel_size": 3},
    {"input_size": (224, 224), "in_channels": 3, "out_channels": 256, "kernel_size": 3},
]

# Calculate and compare results
print(f"{'Type':<12} | {'Params':>8} | {'FLOPs':>12}")
print("-" * 45)
for cfg in configs:
    # Standard convolution
    std_p, std_f = calculate_conv_properties(**cfg, deformable=False)
    
    # DCNv1 calculation
    dcn_p, dcn_f = calculate_conv_properties(**cfg, deformable=True)
    
    # Format values for display
    std_p_k = f"{std_p/1e3:>5.2f}K"
    std_f_m = f"{std_f/1e6:>7.2f}M"
    dcn_p_k = f"{dcn_p/1e3:>5.2f}K"
    dcn_f_m = f"{dcn_f/1e6:>7.2f}M"
    
    print(f"Standard ({cfg['out_channels']}ch) | {std_p_k} | {std_f_m}")
    print(f"DCNv1 ({cfg['out_channels']}ch)   | {dcn_p_k} | {dcn_f_m}")
    print("-" * 45)
