import torch

from ggml_quants_py import ggml_quants


def test_quantize_row_q2k():
    x = torch.randn((1024,))
    qx = ggml_quants.quantize_row_q2_K_impl(x, torch.ones_like(x))
    dq_x = ggml_quants.dequantize_row_q2_K(qx)
    assert dq_x.shape == x.shape
    dq_x = dq_x.reshape(-1, 16)
    x = x.reshape(-1, 16)
    for x_block, block in zip(x, dq_x):
        assert torch.unique(block).numel() <= 4, "Block should be quantized to 2 bits."


def test_quantize_q2k():
    x = torch.randn((4, 1024))
    qx = ggml_quants.quantize_q2_K(x, torch.ones_like(x))
    assert x.size(0) == len(qx)
    assert x.size(1) // ggml_quants.QK_K == len(qx[0])


def test_dequantize_q2k():
    x = torch.randn((4, 1024))
    dqx = ggml_quants.dequantize_q2_K(ggml_quants.quantize_q2_K(x, torch.ones_like(x)))
    assert x.size() == dqx.size()
