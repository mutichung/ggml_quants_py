from dataclasses import dataclass

import torch
from torch import Tensor


QK_K = 256  # TODO: check value; maybe make configurable?
"""Elements per superblock."""
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
        d = self.d.to(dtype=torch.float32)
        dmin = self.dmin.to(dtype=torch.float32)
        scales = self.scales.to(dtype=torch.float32).view(-1, 1)
        mins = self.mins.to(dtype=torch.float32).view(-1, 1)
        qs = self.qs

        dl = d * scales
        ml = dmin * mins
        y = dl * qs - ml
        return y.reshape(-1)


def dequantize_row_q2_K(x: list[SuperBlockQ2K]) -> Tensor:
    """Dequantize a row of weights from Q2K format.

    Args:
        x (list[SuperBlockQ2K]): list of SuperBlockQ2K objects, each representing a superblock.

    Returns:
        Tensor: dequantized tensor in floating point.
    """
    return torch.cat([sb.dequantize() for sb in x], dim=0)


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
            scale, qx, m = make_qkx3_quants(
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
        scale_sb, q_scales = make_qp_quants(15, scales, weight_sums)
        min_sb, q_mins = make_qp_quants(15, mins, weight_sums)
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


def make_qp_quants(
    nmax: int, x: Tensor, quant_weights: Tensor
) -> tuple[Tensor, Tensor]:
    """Quantize a vector x into log_2(nmax + 1) bits.

    Args:
        nmax (int): maximum value for quantization.
        x (torch.FloatTensor): input tensor in floating point.
        quant_weights (Tensor): weights for quantization in floating point.

    Returns:
        tuple[Tensor, Tensor]:
            - scale (Tensor): scaling factor for the quantized values. Scalar, floating point.
            - quantized_x (Tensor): quantized values as integers. Vector of integers.
    """
    x_max = x.max().item()
    if x_max == 0.0:
        return torch.zeros((), dtype=torch.float), torch.zeros_like(
            x, dtype=torch.int
        )  # uint8

    inv_scale = nmax / x_max  # original: iscale
    quantized_x = torch.round(x * inv_scale)
    scale = 1.0 / inv_scale
    best_mse = (quant_weights * (x - quantized_x * scale).square()).sum().item()

    # Try adding small perturbations to the scale to find the best one
    for perturbation in range(-4, 5):
        if perturbation == 0:
            continue
        inv_scale_perturbed = (0.1 * perturbation + nmax) / x_max
        scale_perturbed = 1.0 / inv_scale_perturbed
        quantized_x_perturbed = (
            torch.round(x * inv_scale_perturbed).clamp(min=0, max=nmax).int()
        )
        mse = (
            (quant_weights * (x - quantized_x_perturbed * scale_perturbed).square())
            .sum()
            .item()
        )
        if mse < best_mse:
            best_mse = mse
            inv_scale = inv_scale_perturbed

    # Optimize the quantized values based on the best scale
    quantized_x = torch.round(x * inv_scale).clamp(min=0, max=nmax).int()

    for _ in range(5):
        sumlx = (quant_weights * x * quantized_x).sum()
        suml2 = (quant_weights * quantized_x.square()).sum()

        slx = sumlx - quant_weights * x * quantized_x
        sl2 = suml2 - quant_weights * quantized_x.square()
        new_quants = torch.round(x * sl2 / slx).clamp(min=0, max=nmax).int()
        mask = (slx > 0).logical_and(sl2 > 0.0).logical_not()
        new_quants[mask] = quantized_x[mask]
        mask = new_quants != quantized_x
        slx = slx + quant_weights * x * new_quants
        sl2 = sl2 + quant_weights * new_quants.square()
        if not (slx.square() * suml2 > sumlx.square() * sl2).any():
            break

        quantized_x = new_quants

    sumlx = (quant_weights * x * quantized_x).sum()
    suml2 = (quant_weights * quantized_x.square()).sum()

    return sumlx / suml2, quantized_x


def make_qkx3_quants(
    nmax: int,
    x: Tensor,  # FloatTensor,
    quant_weights: Tensor | None,  # FloatTensor,
    rmin: float,
    rdelta: float,
    nstep: int,
    use_mad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize a vector x into log_2(nmax + 1) bits.

    Args:
        nmax (int): maximum value for quantization.
        x (Tensor): input tensor in floating point.
        quant_weights (Tensor | None): weights for quantization in floating point, if None, use square of x.
        rmin (float):
        rdelta (float):
        nstep (int): number of steps for further optimization.
        use_mad (bool): if True, use mean absolute deviation for optimization, otherwise use mean squared error.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - scale (Tensor): scaling factor for the quantized values. Scalar, floating point.
            - quantized_x (Tensor): quantized values as integers. Vector of integers.
            - xmin (Tensor): minimum value subtracted from x before quantization. Scalar, floating point.
    """
    assert x.ndim == 1
    xmin = x.min().clamp(max=0)
    xmax = x.max()
    if quant_weights is None:
        quant_weights = x.square()
    sum_w = quant_weights.sum()
    sum_x = (quant_weights * x).sum()

    if xmax.item() < xmin.item():
        return torch.zeros(()), torch.zeros_like(x, dtype=torch.int), -xmin

    inv_scale = nmax / (xmax - xmin)
    scale = 1.0 / inv_scale

    quantized_x = torch.round(inv_scale * (x - xmin)).clamp(min=0, max=nmax).int()
    get_metric = torch.abs if use_mad else torch.square
    diff = get_metric(scale * quantized_x + xmin - x)
    best_mad = (quant_weights * diff).sum().item()
    if nstep == 0:
        return scale, quantized_x, -xmin

    for step in range(nstep + 1):
        inv_scale = (rmin + rdelta * step + nmax) / (xmax - xmin)
        qx = torch.round(inv_scale * (x - xmin)).clamp(min=0, max=nmax).int()
        sum_l = (quant_weights * qx).sum()
        sum_l2 = (quant_weights * qx.square()).sum()
        sum_xl = (quant_weights * x * qx).sum()
        D = (sum_w * sum_l2 - sum_l.square()).item()
        if D > 0:
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
            if this_min > 0:
                this_min = torch.zeros(())
                this_scale = sum_xl / sum_l2

            diff = get_metric(this_scale * qx + this_min - x)

            mad = (quant_weights * diff).sum().item()
            if mad < best_mad:
                best_mad = mad
                quantized_x = qx
                scale = this_scale
                xmin = this_min

    return scale, quantized_x, -xmin
