import torch

from editable_gauss_refl.utils.tonemapping import tonemap, untonemap


def test_tonemap():
    x = torch.linspace(0.0, 10.0, 100)
    y = tonemap(x)
    assert y.min() >= 0.0
    assert y.max() <= 1.0
    x_ = untonemap(y)
    assert torch.allclose(x, x_, atol=1e-03)

    y = torch.linspace(0.0, 1.0, 100)
    x = untonemap(y)
    assert x.min() >= 0.0
    y_ = tonemap(x)
    assert torch.allclose(y, y_, atol=1e-05)
