import torch
import torch.nn as nn
import time
import numpy as np
from torchvision.ops import DeformConv2d
from torchinfo import summary

class DepthwiseSeparableDCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, deformable_groups=1, bias=True):
        super(DepthwiseSeparableDCNv2, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.deformable_groups = deformable_groups

        offset_channels = deformable_groups * 2 * kernel_size[0] * kernel_size[1]
        mask_channels   = deformable_groups * 1 * kernel_size[0] * kernel_size[1]
        self.offset_mask_channels = offset_channels + mask_channels

        self.offset_conv = nn.Conv2d(in_channels, self.offset_mask_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation)

        self.depthwise = DeformConv2d(in_channels, in_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation,
                                      groups=in_channels, bias=bias)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.offset_conv(x)
        offset_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        offset = out[:, :offset_channels, :, :]
        mask = out[:, offset_channels:, :, :]
        mask = torch.sigmoid(mask)
        try:
            x = self.depthwise(x, offset, mask)
        except TypeError:
            x = self.depthwise(x, offset)
        return self.pointwise(x)

def analytical_flops(model, output):
    N, C_out, H_out, W_out = output.shape
    output_pixels = H_out * W_out

    offset_weight = model.offset_conv.weight
    in_c_offset = offset_weight.shape[1]
    out_c_offset = offset_weight.shape[0]
    kH_offset, kW_offset = offset_weight.shape[2:]
    flops_offset = 2 * in_c_offset * kH_offset * kW_offset * output_pixels * out_c_offset

    deform_weight = model.depthwise.weight
    in_c_depth = deform_weight.shape[1] * model.depthwise.groups
    out_c_depth = deform_weight.shape[0]
    kH_depth, kW_depth = deform_weight.shape[2:]

    flops_depth = 2 * kH_depth * kW_depth * output_pixels * in_c_depth
    flops_bilinear = 7 * kH_depth * kW_depth * output_pixels * in_c_depth

    pointwise_weight = model.pointwise.weight
    in_c_point = pointwise_weight.shape[1]
    out_c_point = pointwise_weight.shape[0]
    flops_pointwise = 2 * in_c_point * output_pixels * out_c_point

    return flops_offset + flops_depth + flops_bilinear + flops_pointwise

def compare_model(in_shape, out_channels):
    x = torch.randn(1, in_shape[2], in_shape[0], in_shape[1])
    model = DepthwiseSeparableDCNv2(in_shape[2], out_channels, 3, stride=2, padding=1)
    model.eval()

    with torch.no_grad():
        out = model(x)

    emp_summary = summary(model, input_size=(1, in_shape[2], in_shape[0], in_shape[1]), verbose=0)
    emp_params = emp_summary.total_params
    emp_flops = emp_summary.total_mult_adds

    ana_params = sum(p.numel() for p in model.parameters())
    ana_flops = analytical_flops(model, out)

    print(f"--- {in_shape[2]} -> {out_channels} ---")
    print(f"Empirical Params: {emp_params / 1e3:.2f}K | Analytical Params: {ana_params / 1e3:.2f}K | ",
          f"\u0394: {(ana_params - emp_params) / emp_params * 100:.1f}%")
    print(f"Empirical FLOPs: {emp_flops / 1e6:.2f}M | Analytical FLOPs: {ana_flops / 1e6:.2f}M | ",
          f"\u0394: {(ana_flops - emp_flops) / emp_flops * 100:.1f}%")

if __name__ == "__main__":
    in_shape = (224, 224, 3)
    for out_c in [64, 128, 256]:
        compare_model(in_shape, out_c)
