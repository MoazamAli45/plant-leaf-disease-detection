"""Image decoding, preprocessing and the plant-colour check (mirrors the web app)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SIZE = 224


class BadImageError(ValueError):
    """The upload is not a usable image."""


def decode(data: bytes, max_pixels: int) -> Image.Image:
    """Decode bytes into an RGB image. Rejects non-images and decompression bombs."""
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > max_pixels:
            raise BadImageError("image is too large")
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise BadImageError("the file is not a readable image") from e
    return ImageOps.exif_transpose(img).convert("RGB")


def to_input(img: Image.Image) -> np.ndarray:
    """(1, 224, 224, 3) float32 in [0, 255]; the model contains its own preprocessing."""
    resized = img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None]


def plant_pixel_share(img: Image.Image) -> float:
    """Share of saturated pixels with a hue from yellow-brown to green (20-170 degrees).
    Same rule as web/src/lib/verdict.ts."""
    hsv = np.asarray(img.resize((64, 64)).convert("HSV"), dtype=np.float32) / 255.0
    rgb = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255.0
    value = rgb.max(axis=-1)
    chroma = value - rgb.min(axis=-1)
    sat = np.divide(chroma, value, out=np.zeros_like(value), where=value > 0)
    hue = hsv[..., 0] * 360.0
    plant = (value >= 0.12) & (sat >= 0.18) & (hue >= 20) & (hue <= 170)
    return float(plant.mean())


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Heat map (inferno-like colours) blended over the image at 224 x 224."""
    base = img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    c = np.clip(cam, 0.0, 1.0)
    # compact inferno approximation: black -> purple -> orange -> yellow
    stops = np.array(
        [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], float
    )
    pos = c * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    t = (pos - lo)[..., None]
    colours = stops[lo] * (1 - t) + stops[lo + 1] * t
    heat = Image.fromarray(colours.astype(np.uint8))
    return Image.blend(base, heat, alpha)
