from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from . import QK_K
from .ggml_quants import make_qkx2_quants, make_qkx3_quants, make_qp_quants


NUMEL_PER_BLOCK = 32
"""Elements per block."""
BLOCKS_PER_SUPER_BLOCK = QK_K // NUMEL_PER_BLOCK
"""Blocks per superblock. QK_K / 32"""


@dataclass
class SuperBlockQ4K:
    """4-bit K-quantization. Average bpw = 4 + 6 * 2 / 32 + 16 * 2 / 256 = 4.5"""

    d: Tensor
    """Scale for the superblock. Scalar in FP16."""
    dmin: Tensor
    """Min for the superblock. Scalar in FP16."""
    scales: Tensor
    """Scales for each block in the superblock. 6 bits per scale. Integer tensor with shape (8,)."""
    mins: Tensor
    """mins for each block in the superblock. 6 bits per min. Integer tensor with shape (8,)."""
    qs: Tensor
    """Quantized values for the superblock. 4 bits per value. Integer tensor with shape (8, 32)."""

    def dequantize(self) -> Tensor:
        return (
            self.d.float() * self.scales.view(-1, 1) * self.qs
            - self.dmin.float() * self.mins.view(-1, 1)
        ).reshape(-1)


def dequantize_row_q4_K(x: list[SuperBlockQ4K]) -> Tensor:
    return torch.cat([sb.dequantize() for sb in x], dim=0)


def quantize_row_q4_K_ref(x: Tensor):
    assert x.ndim == 1
    assert x.numel() % QK_K == 0

    y: list[SuperBlockQ4K] = []

    x = x.reshape(-1, BLOCKS_PER_SUPER_BLOCK, NUMEL_PER_BLOCK)
    for x_sb in x:
        sum_x2 = x_sb.square().sum(dim=-1)  # (8,)
        av_x = (sum_x2 / NUMEL_PER_BLOCK).sqrt()  # (8,)
        weights = av_x + x_sb.abs()  # (8, 32)
        scales, mins, qx_sb = [], [], []
        for x_block in x_sb:
            scale, qx, min = make_qkx2_quants(
                15, x_block, weights, -1.0, 0.1, 20, False
            )
            scales.append(scale)
            mins.append(min)
            qx_sb.append(qx)

        scales = torch.stack(scales, dim=0)
        mins = torch.stack(mins, dim=0)
        qx_sb = torch.stack(qx_sb, dim=0)

        inv_scale = torch.zeros(())
        inv_min = torch.zeros(())
        if scales.max().item() > 0.0:
            inv_scale = 63.0 / scales.max()
        if mins.max().item() > 0.0:
            inv_min = 63.0 / mins.max()

        scales = (scales * inv_scale).round().clamp(max=63)
        mins = (mins * inv_min).round().clamp(max=63)
        d = (scales.max() / 63.0).half()
        dmin = (mins.max() / 63.0).half()
        qx_sb = (
            torch.round((x_sb + dmin.float() * mins) / (d.float() * scales))
            .clamp(min=0, max=15)
            .int()
        )
        y.append(
            SuperBlockQ4K(
                d=d,
                dmin=dmin,
                scales=scales,
                mins=mins,
                qs=qx_sb,
            )
        )

    return y


def quantize_row_q4_K_impl(
    x: Tensor, quant_weights: Optional[Tensor] = None
) -> list[SuperBlockQ4K]:
    assert x.ndim == 1
    assert x.numel() % QK_K == 0
    assert quant_weights is None or x.size() == quant_weights.size()

    y: list[SuperBlockQ4K] = []

    x = x.reshape(-1, BLOCKS_PER_SUPER_BLOCK, NUMEL_PER_BLOCK)
    if quant_weights is not None:
        quant_weights = quant_weights.reshape(
            -1, BLOCKS_PER_SUPER_BLOCK, NUMEL_PER_BLOCK
        )

    for super_block_idx, x_sb in enumerate(x):
        sum_x2 = x_sb.square().sum()  # scalar
        sigma2 = 2 * sum_x2 / QK_K  # scalar
        if quant_weights is not None:
            qw = quant_weights[super_block_idx]  # (8, 32)
            weights = qw * (sigma2 + x_sb.square())  # (8, 32)
        else:
            weights = sigma2.sqrt() + x_sb.abs()

        sw = weights.sum(dim=-1)  # (8,)
        scales, mins, qx_sb = [], [], []
        for x_block, qw_block in zip(x_sb, weights):
            scale, qx, min = make_qkx3_quants(
                15, x_block, qw_block, -0.9, 0.05, 36, False
            )
            scales.append(scale)
            mins.append(min)
            qx_sb.append(qx)

        scales = torch.stack(scales, dim=0)
        mins = torch.stack(mins, dim=0)
        qx_sb = torch.stack(qx_sb, dim=0)

        d_block, scales = make_qp_quants(63, scales, sw)
        m_block, mins = make_qp_quants(63, mins, sw)

        y.append(
            SuperBlockQ4K(
                d=d_block.half(),
                dmin=m_block.half(),
                scales=scales,
                mins=mins,
                qs=torch.round(
                    (x_sb + m_block.float() * mins) / (d_block.float() * scales)
                ),
            )
        )

    return y


def quantize_q4_K(
    src: Tensor, quant_weights: Optional[Tensor] = None
) -> list[list[SuperBlockQ4K]]:
    assert src.ndim == 2
    if quant_weights is None:
        return [quantize_row_q4_K_ref(x) for x in src]
    else:
        return [
            quantize_row_q4_K_impl(x, qw) for x, qw in zip(src, quant_weights)
        ]
