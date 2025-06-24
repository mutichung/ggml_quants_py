from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from . import QK_K, make_quants

NUMEL_PER_BLOCK = 16
"""16"""
BLOCKS_PER_SUPER_BLOCK = QK_K // NUMEL_PER_BLOCK
"""QK_K / 16"""


@dataclass
class SuperBlockQ2K:
    d: Tensor
    """Scale for the superblock. Scalar in FP16."""
    dmin: Tensor
    """Min for the superblock. Scalar in FP16."""
    scales: Tensor
    """Scales for each block in the superblock. 4 bits per scale. Integer tensor with shape (16,)."""
    mins: Tensor
    """mins for each block in the superblock. 4 bits per min. Integer tensor with shape (16,)."""
    qs: Tensor
    """Quantized values for the superblock. 2 bits per value. Integer tensor with shape (16, 16)."""

    def dequantize(self) -> Tensor:
        """Dequantize the superblock into a floating point tensor.

        Returns:
            Tensor: dequantized tensor in floating point.
        """
        return (
            self.d.float() * self.scales.view(-1, 1) * self.qs
            - self.dmin.float() * self.mins.view(-1, 1)
        ).reshape(-1)


def dequantize_row_q2_K(x: list[SuperBlockQ2K]) -> Tensor:
    """Dequantize a row of weights from Q2K format.

    Args:
        x (list[SuperBlockQ2K]): list of SuperBlockQ2K objects, each representing a superblock.

    Returns:
        Tensor: dequantized tensor in floating point.
    """
    return torch.cat([sb.dequantize() for sb in x], dim=0)


def quantize_row_q2_K_ref(x: Tensor) -> list[SuperBlockQ2K]:
    assert x.ndim == 1
    assert x.numel() % QK_K == 0

    q4scale = torch.tensor(15.0)
    y: list[SuperBlockQ2K] = []

    x = x.reshape(-1, BLOCKS_PER_SUPER_BLOCK, NUMEL_PER_BLOCK)
    for x_sb in x:
        scales, mins, qx_sb = [], [], []
        for x_block in x_sb:
            weights = x_block.abs()
            scale, qx, min = make_quants.make_qkx2_quants(
                3, x_block, weights, -0.5, -0.1, 15, True
            )
            scales.append(scale)
            mins.append(min)
            qx_sb.append(qx)

        scales = torch.stack(scales, dim=0)
        mins = torch.stack(mins, dim=0)
        qx_sb = torch.stack(qx_sb, dim=0)

        # TODO: what's the difference between this and `make_qp_quants`?
        max_scale = scales.max()
        max_min = mins.max()
        if max_scale.item() > 0.0:
            inv_scale = q4scale / max_scale
            scales = torch.round(scales * inv_scale).int()
            mins = torch.round(mins * inv_scale).int()
            d = (max_scale / q4scale).half()
        else:
            scales = torch.zeros_like(scales, dtype=torch.int)
            d = torch.zeros((), dtype=torch.float16)

        if max_min.item() > 0.0:
            inv_scale = q4scale / max_min
            mins = torch.round(mins * inv_scale).int()
            dmin = (max_min / q4scale).half()
        else:
            mins = torch.zeros_like(mins, dtype=torch.int)
            dmin = torch.zeros((), dtype=torch.float16)

        dq_scales = (d * scales).float()
        dq_mins = (dmin * mins).float()
        qx_sb = torch.round((x_sb + dq_mins) / dq_scales).clamp(min=0, max=3).int()

        y.append(
            SuperBlockQ2K(
                d=d,
                dmin=dmin,
                scales=scales,
                mins=mins,
                qs=qx_sb,
            )
        )

    return y


def quantize_row_q2_K_impl(x: Tensor, quant_weights: Tensor) -> list[SuperBlockQ2K]:
    """Quantize a row of weights into Q2K format.

    Args:
        x (Tensor): input tensor in floating point.
        quant_weights (Tensor): weights for quantization in floating point.

    Returns:
        list[SuperBlockQ2K]: list of SuperBlockQ2K objects, each representing a superblock.
    """
    assert x.ndim == 1
    assert x.numel() % QK_K == 0
    assert x.size() == quant_weights.size()
    requantize: bool = True

    # L: list[int] = [0] * QK_K
    # Laux: list[int] = [0] * NUMEL_PER_BLOCK
    # mins: list[float] = [0.0] * BLOCKS_PER_SUPER_BLOCK
    # scales: list[float] = [0.0] * BLOCKS_PER_SUPER_BLOCK
    # weight_sums: list[float] = [0.0] * BLOCKS_PER_SUPER_BLOCK  # original: sw
    # weight: list[float] = [0.0] * NUMEL_PER_BLOCK  # no need to initialize
    # Ls: list[int] = [0] * BLOCKS_PER_SUPER_BLOCK
    # Lm: list[int] = [0] * BLOCKS_PER_SUPER_BLOCK

    y: list[SuperBlockQ2K] = []

    x = x.reshape(-1, BLOCKS_PER_SUPER_BLOCK, NUMEL_PER_BLOCK)
    quant_weights = quant_weights.reshape(-1, BLOCKS_PER_SUPER_BLOCK, NUMEL_PER_BLOCK)

    for x_sb, qw_sb in zip(x, quant_weights):
        # x_sb: (16, 16); qw_sb: (16, 16)
        sum_x2 = x_sb.square().sum()
        sigma2 = sum_x2 / QK_K
        weight_sums = (qw_sb * torch.sqrt(sigma2 + x_sb.square())).sum(dim=-1)

        scales, mins, qx_sb = [], [], []
        for x_block, qw_block in zip(x_sb, qw_sb):
            scale, qx, m = make_quants.make_qkx3_quants(
                3,
                x_block,
                qw_block,
                -0.9,
                0.05,
                36,
                False,
            )
            scales.append(scale)
            mins.append(m)
            qx_sb.append(qx)

        scales = torch.stack(scales, dim=0)
        mins = torch.stack(mins, dim=0)
        qx_sb = torch.stack(qx_sb, dim=0)

        # Quantize block scales and mins.
        scale_sb, q_scales = make_quants.make_qp_quants(15, scales, weight_sums)
        min_sb, q_mins = make_quants.make_qp_quants(15, mins, weight_sums)
        assert scale_sb.dim() == 0 and min_sb.dim() == 0, (
            "scale_sb and min_sb should be scalars"
        )

        curr_y = SuperBlockQ2K(
            d=scale_sb.to(dtype=torch.float16),
            dmin=min_sb.to(dtype=torch.float16),
            scales=q_scales,
            mins=q_mins,
            qs=qx_sb,
        )

        if requantize:
            d = scale_sb * curr_y.scales
            m = min_sb * curr_y.mins
            curr_y.qs = torch.round((x_sb + m) / d).clamp(min=0, max=3).int()

        y.append(curr_y)

    return y


def quantize_q2_K(
    src: Tensor, quant_weights: Optional[Tensor] = None
) -> list[list[SuperBlockQ2K]]:
    assert src.ndim == 2
    if quant_weights is None:
        return [quantize_row_q2_K_ref(x) for x in src]
    else:
        return [quantize_row_q2_K_impl(x, qw) for x, qw in zip(src, quant_weights)]


def dequantize_q2_K(src: list[list[SuperBlockQ2K]]) -> Tensor:
    return torch.stack([dequantize_row_q2_K(x) for x in src])
