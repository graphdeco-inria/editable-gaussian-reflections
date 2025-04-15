import numpy as np
import torch

# exposure = 3.5
gamma = 1.3


def tonemap(x):  # filmic tonemapping
    return ((x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)) ** gamma


def untonemap(y):
    _sqrt = np.sqrt if isinstance(y, np.ndarray) else torch.sqrt
    y = y ** (1 / gamma)
    numerator = (-0.1371 * y) - (0.09549 * _sqrt(y**2 - 0.1512 * y + 0.1783)) + 0.04032
    denominator = y - 1
    x = numerator / denominator
    return x
