import torch

from ggml_quants_py.k_quant import q2k


def test_quantize_row_q2k_ref():
    x = torch.randn((1024,))
    qx = q2k.quantize_row_q2_K_ref(x)
    dq_x = q2k.dequantize_row_q2_K(qx)
    assert dq_x.shape == x.shape
    dq_x = dq_x.reshape(-1, 16)
    x = x.reshape(-1, 16)
    for x_block, block in zip(x, dq_x):
        assert torch.unique(block).numel() <= 4, "Block should be quantized to 2 bits."


def test_quantize_row_q2k_impl():
    x = torch.randn((1024,))
    qx = q2k.quantize_row_q2_K_impl(x, torch.ones_like(x))
    dq_x = q2k.dequantize_row_q2_K(qx)
    assert dq_x.shape == x.shape
    dq_x = dq_x.reshape(-1, 16)
    x = x.reshape(-1, 16)
    for x_block, block in zip(x, dq_x):
        assert torch.unique(block).numel() <= 4, "Block should be quantized to 2 bits."


def test_quantize_q2k():
    x = torch.randn((4, 1024))
    qx = q2k.quantize_q2_K(x)
    assert x.size(0) == len(qx)
    assert x.size(1) // q2k.QK_K == len(qx[0])


def test_quantize_q2k_with_weights():
    x = torch.randn((4, 1024))
    qx = q2k.quantize_q2_K(x, torch.ones_like(x))
    assert x.size(0) == len(qx)
    assert x.size(1) // q2k.QK_K == len(qx[0])


def test_dequantize_q2k():
    x = torch.randn((4, 1024))
    dqx = q2k.dequantize_q2_K(q2k.quantize_q2_K(x, torch.ones_like(x)))
    assert x.size() == dqx.size()
