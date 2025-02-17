import numpy as np
import torch

def tonemap(x, exposure=2.2):
    x *= exposure
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)

def untonemap(y, exposure=2.2):
    _sqrt = np.sqrt if isinstance(y, np.ndarray) else torch.sqrt
    numerator = (-0.1371 * y) - (0.09549 * _sqrt(y**2 - 0.1512 * y + 0.1783)) + 0.04032
    denominator = y - 1
    x = numerator / denominator
    return x / exposure