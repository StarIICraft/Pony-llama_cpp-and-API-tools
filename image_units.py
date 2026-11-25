PIXELS_PER_MEGAPIXEL = 1024 * 1024


def megapixels_to_pixels(megapixels):
    return max(
        1,
        int(float(megapixels) * PIXELS_PER_MEGAPIXEL),
    )


__all__ = [
    "PIXELS_PER_MEGAPIXEL",
    "megapixels_to_pixels",
]
