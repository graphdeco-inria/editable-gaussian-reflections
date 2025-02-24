import torch 
import kornia
from torchvision.utils import save_image
import torch.nn.functional as F 
# gaussian filter

image = torch.randn(1, 3, 512, 768).cuda()

def adaptive_gaussian_blur(image: torch.Tensor, sigma: float):
    """
    Applies Gaussian blur efficiently by automatically selecting a downsampling factor 
    and kernel size based on sigma, applying the blur at a lower resolution, and then upsampling back.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        torch.Tensor: Blurred image of the same shape as input.
    """
    # Automatically determine kernel size (ensure it's odd)
    kernel_size = max(3, int(2 * round(3 * sigma) + 1))

    # Determine downsampling factor based on sigma
    downscale_factor = max(1, int(sigma // 2))
    downscale_factor = 2 ** (downscale_factor.bit_length() - 1)
    # Ensure the image doesn't go smaller than 128 pixels in any dimension
    min_size = min(image.shape[-2:])
    while min_size // downscale_factor < 128:
        downscale_factor //= 2

    if downscale_factor > 1:
        # Downsample
        small_image = F.interpolate(image, scale_factor=1/downscale_factor, mode='bilinear', align_corners=False)

        # Apply Gaussian blur at lower resolution
        print(small_image.shape)
        blurred_small = kornia.filters.gaussian_blur2d(small_image, (kernel_size, kernel_size), (sigma, sigma))

        # Upsample back to original size
        blurred_image = F.interpolate(blurred_small, size=image.shape[-2:], mode='bilinear', align_corners=False)
    else:
        # Directly apply Gaussian blur if downsampling isn't needed
        blurred_image = kornia.filters.gaussian_blur2d(image, (kernel_size, kernel_size), (sigma, sigma))

    return blurred_image


# blur the image
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

for sigma in [1e-20, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    kernel_size = 2 * int(3 * sigma) + 1
    start_event.record()
    image_blurred = kornia.filters.gaussian_blur2d(image, (kernel_size, kernel_size), (sigma, sigma))
    # image_blurred = adaptive_gaussian_blur(image, sigma)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time} ms")


    save_image(image_blurred, 'blurred_image.png')
    breakpoint()