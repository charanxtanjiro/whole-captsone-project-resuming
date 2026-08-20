from PIL import Image
import numpy as np


def pixelate_image(input_path: str, output_path: str, pixel_size: int = 8) -> Image.Image:
    """Create a pixelated version of the image and save it.

    Args:
        input_path: Path to source image.
        output_path: Path where pixelated image will be saved.
        pixel_size: Size of the pixel blocks (larger => more pixelated).

    Returns:
        The pixelated PIL Image object.
    """
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    # Downscale then upscale with nearest neighbor to create pixel effect
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    small = img.resize((small_w, small_h), resample=Image.NEAREST)
    result = small.resize((w, h), Image.NEAREST)
    result.save(output_path)
    return result


def image_to_pixel_array(input_path: str, pixel_size: int = 1) -> np.ndarray:
    """Return a 2D array of pixels (height x width x 3) for the image.

    If `pixel_size` > 1 the image will be sampled as blocks of that size
    (useful to obtain a reduced-resolution pixel grid).
    """
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    if pixel_size > 1:
        small_w = max(1, w // pixel_size)
        small_h = max(1, h // pixel_size)
        img = img.resize((small_w, small_h), resample=Image.NEAREST)
    arr = np.array(img)
    return arr


if __name__ == "__main__":
    # Quick demo when run directly
    import sys
    if len(sys.argv) < 3:
        print("Usage: python image_pixel_converter.py <input> <output> [pixel_size]")
    else:
        inp = sys.argv[1]
        out = sys.argv[2]
        size = int(sys.argv[3]) if len(sys.argv) >= 4 else 8
        pixelate_image(inp, out, size)
        print(f"Saved pixelated image to {out}")
