import torch

from ggml_quants_py.k_quant import make_quants


def test_make_qx_quants():
    x = torch.randn((16,))
    nmax = 2**15
    scale, qx = make_quants.make_qx_quants(nmax, x, 3)
    assert qx.shape == x.shape
    dqx = (qx - nmax) * scale
    torch.testing.assert_close(x, dqx, rtol=1e-3, atol=1e-3)
