import torch
from torch import Tensor

from . import QK_K
from .ggml_quants import make_qkx2_quants


NUMEL_PER_BLOCK = 32
"""Elements per block."""
BLOCKS_PER_SUPER_BLOCK = QK_K // NUMEL_PER_BLOCK
"""Blocks per superblock. QK_K / 32"""


def quantize_row_q4_K_ref(x: Tensor):
    assert x.ndim == 1
    assert x.numel() % QK_K == 0

    x = x.reshape(-1, BLOCKS_PER_SUPER_BLOCK, NUMEL_PER_BLOCK)
    for x_sb in x:
        sum_x2 = x_sb.square().sum(dim=-1)  # (8,)
        av_x = (sum_x2 / NUMEL_PER_BLOCK).sqrt()  # (8,)
        weights = av_x + x_sb.abs()  # (8, 32)
        scales, mins, qx_sb = [], [], []
        for x_block in x_sb:
            scale, qx, min = make_qkx2_quants(15, x_block, weights, -1.0, 0.1, 20, False)
            scales.append(scale)
            mins.append(min)
            qx_sb.append(qx)

        scales = torch.stack(scales, dim=0)
        mins = torch.stack(mins, dim=0)
        qx_sb = torch.stack(qx_sb, dim=0)

        max_scale = scales.max()
        max_min = mins.max()

        if max_scale.item() > 0.0:
            inv_scale = 63.0 / max_scale
        else:
            inv_scale = torch.zeros(())

        if max_min.item() > 0.0:
            inv_min = 63.0 / max_min
        else:
            inv_min = torch.zeros(())


