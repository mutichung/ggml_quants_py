import torch

from ggml_quants_py import q4k, QK_K


def test_quantize_row_q4k_ref():
    x = torch.randn((1024,))
    qx = q4k.quantize_row_q4_K_ref(x)
    dq_x = q4k.dequantize_row_q4_K(qx)
    assert dq_x.shape == x.shape
    dq_x = dq_x.reshape(-1, 16)
    x = x.reshape(-1, 16)
    for x_block, block in zip(x, dq_x):
        assert torch.unique(block).numel() <= 16, "Block should be quantized to 4 bits."


def test_quantize_row_q4k_impl():
    x = torch.randn((1024,))
    qx = q4k.quantize_row_q4_K_impl(x, torch.ones_like(x))
    dq_x = q4k.dequantize_row_q4_K(qx)
    assert dq_x.shape == x.shape
    dq_x = dq_x.reshape(-1, 16)
    x = x.reshape(-1, 16)
    for x_block, block in zip(x, dq_x):
        assert torch.unique(block).numel() <= 16, "Block should be quantized to 4 bits."


def test_quantize_q4k():
    x = torch.randn((4, 1024))
    qx = q4k.quantize_q4_K(x)
    assert x.size(0) == len(qx)
    assert x.size(1) // QK_K == len(qx[0])


def test_quantize_q4k_with_weights():
    x = torch.randn((4, 1024))
    qx = q4k.quantize_q4_K(x, torch.ones_like(x))
    assert x.size(0) == len(qx)
    assert x.size(1) // QK_K == len(qx[0])


def test_dequantize_q4k():
    x = torch.randn((4, 1024))
    dqx = q4k.dequantize_q4_K(q4k.quantize_q4_K(x, torch.ones_like(x)))
    assert x.size() == dqx.size()
