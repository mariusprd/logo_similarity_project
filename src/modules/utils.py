# src/modules/utils.py (New File)

from imagehash import hex_to_hash
from colormath.color_objects import sRGBColor
from colormath.color_diff import delta_e_cie2000
import numpy as np

def calculate_phash_similarity(hash1: str, hash2: str) -> float:
    """
    Calculates similarity based on hamming distance of two pHashes.
    Similarity is 1 - (distance / max_distance).
    """
    if not hash1 or not hash2:
        return 0.0

    try:
        distance = hex_to_hash(hash1) - hex_to_hash(hash2)
        max_distance = 64  # pHash produces a 64-bit hash
        return 1.0 - (distance / max_distance)
    except Exception:
        return 0.0

def _hex_to_rgb(hex_color: str) -> sRGBColor:
    """Converts a hex color string to an sRGBColor object."""
    hex_color = hex_color.lstrip('#')
    return sRGBColor.new_from_rgb_hex(hex_color)

def calculate_palette_similarity(palette1: list, palette2: list) -> float:
    """
    Calculates the similarity between two color palettes.
    Each palette item is expected to be a tuple: ((r, g, b), confidence).
    """
    if not palette1 or not palette2:
        return 0.0

    try:
        # Extract just the RGB tuples
        rgb_palette1 = [sRGBColor(r, g, b, is_upscaled=True) for (r, g, b), _ in palette1]
        rgb_palette2 = [sRGBColor(r, g, b, is_upscaled=True) for (r, g, b), _ in palette2]

        # Convert to Lab color space for accurate difference calculation
        lab_palette1 = [c.convert_to('lab') for c in rgb_palette1]
        lab_palette2 = [c.convert_to('lab') for c in rgb_palette2]

        # Find the minimum color difference for each color in palette1 to palette2
        total_min_diff = 0
        for color1 in lab_palette1:
            min_diff = min([delta_e_cie2000(color1, color2) for color2 in lab_palette2])
            total_min_diff += min_diff

        avg_diff = total_min_diff / len(lab_palette1)

        # Normalize the difference to a 0-1 similarity score
        # (100 is a very large difference, so this is a simple normalization)
        return max(0, 1 - avg_diff / 100.0)
    except Exception:
        return 0.0