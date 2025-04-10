import torch
from torch import nn
from torchvision.ops import deform_conv2d
from torchinfo import summary
import math
import pandas as pd

class DeformDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size**2, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.mask_conv = nn.Conv2d(
            in_channels, kernel_size**2, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.depthwise_weights = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size, kernel_size)
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        outputs = []
        for c in range(self.in_channels):
            x_c = x[:, c:c+1, :, :]
            w_c = self.depthwise_weights[c:c+1]
            out_c = deform_conv2d(x_c, offset, w_c, mask=mask, padding=self.kernel_size // 2)
            outputs.append(out_c)
        out = torch.cat(outputs, dim=1)
        return self.pointwise(out)

def analytical_flops_params(in_ch, out_ch, input_size, kernel_size=3):
    H, W = input_size
    K = kernel_size**2

    offset_params = (in_ch * K + 1) * 2 * K
    mask_params = (in_ch * K + 1) * K
    depthwise_params = in_ch * K
    pointwise_params = in_ch * out_ch + out_ch

    offset_flops = 2 * in_ch * K * (2*K) * H * W
    mask_flops = 2 * in_ch * K * K * H * W
    depthwise_flops = in_ch * K * 4 * H * W  # 4 ops per kernel point
    pointwise_flops = 2 * in_ch * out_ch * H * W

    total_params = offset_params + mask_params + depthwise_params + pointwise_params
    total_flops = offset_flops + mask_flops + depthwise_flops + pointwise_flops

    return total_params, total_flops

# -------------------------------
# Main Comparison Loop
# -------------------------------
input_size = (64, 64)  # Reduce size to avoid OOM
configs = [(3, 64), (3, 128), (3, 256)]
results = []

for in_ch, out_ch in configs:
    print(f"\n--- {in_ch} -> {out_ch} ---")
    model = DeformDepthwiseSeparableConv(in_ch, out_ch)
    summary_stats = summary(model, input_size=(1, in_ch, *input_size), verbose=0)
    
    emp_params = sum(p.numel() for p in model.parameters())
    emp_flops = summary_stats.total_mult_adds

    ana_params, ana_flops = analytical_flops_params(in_ch, out_ch, input_size)

    delta_params = 100 * (emp_params - ana_params) / ana_params
    delta_flops = 100 * (emp_flops - ana_flops) / ana_flops

    print(f"Empirical Params: {emp_params/1e3:.2f}K | Analytical Params: {ana_params/1e3:.2f}K | Δ: {delta_params:.1f}%")
    print(f"Empirical FLOPs: {emp_flops/1e6:.2f}M | Analytical FLOPs: {ana_flops/1e6:.2f}M | Δ: {delta_flops:.1f}%")

    results.append({
        "Channels": f"{in_ch}->{out_ch}",
        "Emp Params": emp_params / 1e3,
        "Ana Params": ana_params / 1e3,
        "ΔParams (%)": delta_params,
        "Emp FLOPs": emp_flops / 1e6,
        "Ana FLOPs": ana_flops / 1e6,
        "ΔFLOPs (%)": delta_flops,
    })

df = pd.DataFrame(results)
print("\nFinal Comparison Table:")
print(df.round(2).to_string(index=False))

