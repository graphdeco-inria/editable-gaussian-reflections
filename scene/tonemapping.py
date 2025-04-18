def tonemap(x, gamma=1.3):  # filmic tonemapping
    x = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
    x = x**gamma
    return x


def untonemap(y, gamma=1.3, eps=1e-6):
    y = y ** (1 / gamma)
    numerator = 0.1371 * y + 0.09549 * (y**2 - 0.1512 * y + 0.1783) ** 0.5 - 0.04032
    denominator = 1 - y + eps
    x = numerator / denominator
    return x
