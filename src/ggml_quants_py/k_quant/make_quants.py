from typing import Optional

import torch
from torch import Tensor

GROUP_MAX_EPS = 1e-15


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


def make_qkx2_quants(
    nmax: int,
    x: Tensor,
    quant_weights: Tensor,
    rmin: float,
    rdelta: float,
    nstep: int,
    use_mad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize a vector x into log_2(nmax + 1) bits.

    Args:
        nmax (int): maximum value for quantization.
        x (Tensor): input tensor in floating point.
        quant_weights (Tensor): weights for quantization in floating point.
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
    xmin = x.min().clamp(max=0.0)
    xmax = x.max()
    sum_w = quant_weights.sum()
    sum_x = (quant_weights * x).sum()

    if xmax.item() == xmin.item():
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


def make_qkx3_quants(
    nmax: int,
    x: Tensor,
    quant_weights: Tensor | None,
    rmin: float,
    rdelta: float,
    nstep: int,
    use_mad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if quant_weights is None:
        quant_weights = x.square()
    return make_qkx2_quants(nmax, x, quant_weights, rmin, rdelta, nstep, use_mad)


def make_qx_quants(
    nmax: int, x: Tensor, rmse_type: int, qw: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """Symmetric quantization of vector x into log_2(2 * nmax) bits.

    Args:
        nmax (int): maximum absolute value for quantization.
        x (Tensor): input tensor in floating point.
        rmse_type (int): type of error metric.
        qw (Optional[Tensor], optional): quantization weights. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]:
            - Scale (Tensor): scaling factor for the quantized values. Scalar, FP32.
            - quantized_x (Tensor): unsigned integer vector with range [0, 2 * nmax - 1].
    """
    assert x.ndim == 1
    max_idx = x.abs().argmax()
    xmax = x[max_idx]
    amax = xmax.abs()

    if amax.item() < GROUP_MAX_EPS:
        return torch.zeros(()), torch.zeros_like(x, dtype=torch.int)

    inv_scale = -nmax / xmax
    if rmse_type == 0:
        quantized_x = torch.round(x * inv_scale).clamp(min=-nmax, max=nmax - 1).int()
        return 1.0 / inv_scale, quantized_x

    return_early = False
    if rmse_type < 0:
        rmse_type = -rmse_type
        return_early = True

    qx_sym = torch.round(x * inv_scale).clamp(min=-nmax, max=nmax - 1).int()
    quantized_x = qx_sym + nmax

    def get_quant_weight(qw: Tensor | None, rmse_type: int) -> Tensor:
        if qw is not None:
            return qw
        if rmse_type == 1:
            return x.square()
        if rmse_type == 2:
            return torch.ones_like(x)
        if rmse_type == 3:
            return x.abs()
        return x.abs().sqrt()

    w = get_quant_weight(qw, rmse_type)
    sumlx = (w * x * qx_sym).sum()
    suml2 = (w * qx_sym.square()).sum()
    if suml2 > 0:
        scale = sumlx / suml2
    else:
        scale = torch.zeros(())

    if return_early:
        if suml2 > 0:
            return 0.5 * (scale + 1 / inv_scale), quantized_x
        else:
            return 1 / inv_scale, quantized_x

    best = scale * sumlx
    for dscale in range(-9, 10):
        if dscale == 0:
            continue
        inv_scale = -(nmax + 0.1 * dscale) / xmax
        qx_sym = torch.round(inv_scale * x).clamp(min=-nmax, max=nmax - 1)
        w = get_quant_weight(qw, rmse_type)
        sumlx = (w * x * qx_sym).sum()
        suml2 = (w * qx_sym * qx_sym).sum()
        if suml2 > 0 and sumlx.square() > best * suml2:
            qx_sym = torch.round(inv_scale * x).clamp(min=-nmax, max=nmax - 1)
            quantized_x = qx_sym + nmax
            scale = sumlx / suml2
            best = scale * sumlx

    return scale, quantized_x
