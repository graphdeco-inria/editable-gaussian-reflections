import numpy as np
from PIL import Image


def to_pil_image(arr: np.ndarray) -> Image.Image:
    _, _, c = arr.shape
    if c == 3:
        img = Image.fromarray((arr * (2**8 - 1)).round().astype(np.uint8))
    elif c == 1:
        img = Image.fromarray((arr[:, :, 0] * (2**16 - 1)).round().astype(np.uint16))
    else:
        raise ValueError(f"Number of channels not supported")
    return img


def from_pil_image(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr / (2**8 - 1)
    elif arr.ndim == 2:
        arr = arr / (2**16 - 1)
        arr = arr[:, :, None]
    return arr
