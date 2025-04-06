import math

def dcnv3_properties(input_size, in_channels, out_channels, kernel_size, groups=4):
    """
    Calculates parameters and FLOPs for DCNv3 with:
    - Spatial-grouped modulation
    - Softmax normalization
    - Optimized memory access patterns
    """
    H, W = input_size
    K = kernel_size ** 2  # Sampling points
    
    # Base convolution components
    std_params = (kernel_size**2 * in_channels + 1) * out_channels
    std_flops = 2 * (kernel_size**2 * in_channels) * out_channels * H * W

    # Dynamic components (offset + modulation + grouping)
    group_dim = out_channels // groups
    offset_mod_params = (kernel_size**2 * in_channels + 1) * 3 * K * groups
    
    # Offset/modulation prediction FLOPs (group-wise processing)
    offset_mod_flops = 2 * (kernel_size**2 * in_channels) * 3 * K * groups * H * W

    # Deformable operations (interpolation + modulation + softmax)
    interp_flops = K * 5 * H * W * out_channels  # 4 interp + 1 modulation
    softmax_flops = 3 * K * H * W * out_channels  # Softmax normalization

    total_params = std_params + offset_mod_params
    total_flops = std_flops + offset_mod_flops + interp_flops + softmax_flops

    return total_params, total_flops

def print_dcnv3_analysis(configs):
    print(f"{'Type':<20} | {'Params':>10} | {'FLOPs':>12}")
    print("-" * 45)
    
    for cfg in configs:
        # Standard convolution
        std_p = (cfg['kernel_size']**2 * cfg['in_channels'] + 1) * cfg['out_channels']
        std_f = 2 * (cfg['kernel_size']**2 * cfg['in_channels']) * cfg['out_channels'] * cfg['input_size'][0] * cfg['input_size'][1]
        
        # DCNv3 calculation
        dcn3_p, dcn3_f = dcnv3_properties(**cfg)

        # Formatting values
        std_p_str = f"{std_p/1e3:>8.2f}K"
        std_f_str = f"{std_f/1e6:>9.2f}M"
        dcn3_p_str = f"{dcn3_p/1e3:>8.2f}K"
        dcn3_f_str = f"{dcn3_f/1e6:>9.2f}M"

        print(f"Standard ({cfg['out_channels']}ch)  | {std_p_str} | {std_f_str}")
        print(f"DCNv3 ({cfg['out_channels']}ch)    | {dcn3_p_str} | {dcn3_f_str}")
        print("-" * 45)

# Configuration from research papers
configurations = [
    {'input_size': (224, 224), 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3},
    {'input_size': (224, 224), 'in_channels': 3, 'out_channels': 128, 'kernel_size': 3},
    {'input_size': (224, 224), 'in_channels': 3, 'out_channels': 256, 'kernel_size': 3},
]

print_dcnv3_analysis(configurations)
