"""ASCII art generation module ported from TerminalImageViewer.

This module provides functions to convert images to ASCII/Unicode art using
block characters and ANSI color codes.
"""

from typing import Callable, NamedTuple, Tuple, List, Optional, Dict, TYPE_CHECKING
import math
from collections import defaultdict

from ansi_image.tables import (
    FLAG_FG, FLAG_BG, FLAG_MODE_256, FLAG_24BIT, FLAG_NOOPT, FLAG_TELETEXT,
    COLOR_STEP_COUNT, COLOR_STEPS, GRAYSCALE_STEP_COUNT, GRAYSCALE_STEPS,
    END_MARKER, BITMAPS
)

from PIL import Image

# Type alias for pixel getter function
GetPixelFunction = Callable[[int, int], int]


def get_channel(rgb: int, index: int) -> int:
    """Extract a color channel (R, G, or B) from an RGB value.

    Args:
        rgb: RGB color value as an integer (0xRRGGBB format)
        index: Channel index (0=R, 1=G, 2=B)

    Returns:
        The value of the specified color channel (0-255)
    """
    return (rgb >> ((2 - index) * 8)) & 255


def best_index(value: int, steps: List[int]) -> int:
    """Find the index of the closest value in a STEPS array to a given value.

    This function finds the index in the steps array whose value is closest
    to the given value, using absolute difference as the distance metric.

    Args:
        value: The target value to find the closest match for
        steps: List of step values to search through

    Returns:
        The index of the closest value in the steps array

    Raises:
        ValueError: If steps array is empty
    """
    if not steps:
        raise ValueError("Steps array cannot be empty")

    best_diff = abs(steps[0] - value)
    result = 0

    for i in range(1, len(steps)):
        diff = abs(steps[i] - value)
        if diff < best_diff:
            result = i
            best_diff = diff

    return result


def clamp_byte(value: int) -> int:
    """Clamp a value to the 0-255 byte range.

    Args:
        value: The value to clamp

    Returns:
        The clamped value in the range [0, 255]
    """
    return 0 if value < 0 else (255 if value > 255 else value)


def sqr(n: float) -> float:
    """Square a number.

    Args:
        n: The number to square.

    Returns:
        The square of the input number.
    """
    return n * n


def print_term_color(flags: int, r: int, g: int, b: int) -> str:
    """Generate ANSI color codes for terminal colors.

    This function outputs ANSI color codes for terminal colors, supporting both
    24-bit true color and 256-color modes. It handles both foreground and
    background colors based on flags.

    Args:
        flags: Bit flags controlling color mode and type
               - FLAG_BG: Use background color instead of foreground
               - FLAG_MODE_256: Use 256-color mode instead of 24-bit true color
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)

    Returns:
        ANSI escape sequence string for the specified color
    """
    r = clamp_byte(r)
    g = clamp_byte(g)
    b = clamp_byte(b)

    bg = (flags & FLAG_BG) != 0

    # 24-bit true color mode (default)
    if (flags & FLAG_MODE_256) == 0:
        if bg:
            return f"\x1b[48;2;{r};{g};{b}m"
        else:
            return f"\x1b[38;2;{r};{g};{b}m"

    # 256-color mode - find best color approximation
    ri = best_index(r, COLOR_STEPS)
    gi = best_index(g, COLOR_STEPS)
    bi = best_index(b, COLOR_STEPS)

    rq = COLOR_STEPS[ri]
    gq = COLOR_STEPS[gi]
    bq = COLOR_STEPS[bi]

    # Calculate grayscale equivalent using standard luminance weights
    gray = round(r * 0.2989 + g * 0.5870 + b * 0.1140)

    gri = best_index(gray, GRAYSCALE_STEPS)
    grq = GRAYSCALE_STEPS[gri]

    # Calculate weighted squared distances to determine if color or grayscale is better
    # Use standard RGB to luminance weights for distance calculation
    color_distance = 0.3 * sqr(rq - r) + 0.59 * sqr(gq - g) + 0.11 * sqr(bq - b)
    gray_distance = 0.3 * sqr(grq - r) + 0.59 * sqr(grq - g) + 0.11 * sqr(grq - b)

    if color_distance < gray_distance:
        # Use color palette (216 colors: 6x6x6 RGB cube starting at index 16)
        color_index = 16 + 36 * ri + 6 * gi + bi
    else:
        # Use grayscale palette (24 grays starting at index 232)
        color_index = 232 + gri

    if bg:
        return f"\x1b[48;5;{color_index}m"
    else:
        return f"\x1b[38;5;{color_index}m"


class CharData:
    """Character data for ASCII art rendering.

    Attributes:
        fg_color: Foreground color as [R, G, B] list
        bg_color: Background color as [R, G, B] list
        codepoint: Unicode codepoint for the character
    """

    def __init__(
        self,
        fg_color: Optional[List[int]] = None,
        bg_color: Optional[List[int]] = None,
        codepoint: int = 0x2584,
    ):
        self.fg_color = fg_color if fg_color is not None else [0, 0, 0]
        self.bg_color = bg_color if bg_color is not None else [0, 0, 0]
        self.codepoint = codepoint

    def __eq__(self, other: object) -> bool:
        """Check equality between CharData instances."""
        if not isinstance(other, CharData):
            return False
        return (
            self.fg_color == other.fg_color
            and self.bg_color == other.bg_color
            and self.codepoint == other.codepoint
        )

    def __repr__(self) -> str:
        """String representation of CharData."""
        return f"CharData(fg_color={self.fg_color}, bg_color={self.bg_color}, codepoint=0x{self.codepoint:x})"


def create_char_data(
    get_pixel: GetPixelFunction, x0: int, y0: int, codepoint: int, pattern: int
) -> CharData:
    """Create CharData with average foreground and background colors.

    This function analyzes a 4x8 pixel area and computes average colors for
    foreground and background pixels based on the given pattern. The pattern
    is a 32-bit mask where each bit corresponds to a pixel in the 4x8 area,
    with the most significant bit representing the top-left pixel.

    Args:
        get_pixel: Function that returns RGB color for pixel at (x, y)
        x0: Starting x coordinate of the 4x8 pixel area
        y0: Starting y coordinate of the 4x8 pixel area
        codepoint: Unicode codepoint for this character
        pattern: 32-bit pattern where 1=foreground, 0=background

    Returns:
        CharData with computed average foreground/background colors
    """
    result = CharData()
    result.codepoint = codepoint

    fg_count = 0
    bg_count = 0
    mask = 0x80000000  # Start with most significant bit

    # Process 4x8 pixel area (32 pixels total)
    for y in range(8):
        for x in range(4):
            # Determine if this pixel is foreground or background
            if pattern & mask:
                # Foreground pixel
                target_color = result.fg_color
                fg_count += 1
            else:
                # Background pixel
                target_color = result.bg_color
                bg_count += 1

            rgb = get_pixel(x0 + x, y0 + y)

            for i in range(3):
                target_color[i] += get_channel(rgb, i)

            mask = mask >> 1

    for i in range(3):
        if bg_count != 0:
            result.bg_color[i] //= bg_count
        if fg_count != 0:
            result.fg_color[i] //= fg_count

    return result


def print_codepoint(codepoint: int) -> str:
    """Convert a Unicode codepoint to UTF-8 bytes and return as string.

    This function converts Unicode codepoints to UTF-8 encoding and returns
    the UTF-8 bytes as a string. It handles different UTF-8 encoding ranges:
    - 1-byte (ASCII): U+0000 to U+007F
    - 2-byte: U+0080 to U+07FF
    - 3-byte: U+0800 to U+FFFF
    - 4-byte: U+10000 to U+10FFFF

    Args:
        codepoint: Unicode codepoint as an integer

    Returns:
        UTF-8 encoded string representation of the codepoint

    Raises:
        ValueError: If codepoint is outside valid Unicode range (0 to 0x10FFFF)
    """
    if codepoint < 0:
        raise ValueError(f"Invalid codepoint: {codepoint} (must be non-negative)")
    elif codepoint < 128:
        # 1-byte UTF-8 (ASCII)
        return chr(codepoint)
    elif codepoint <= 0x7FF:
        # 2-byte UTF-8
        byte1 = 0xC0 | (codepoint >> 6)
        byte2 = 0x80 | (codepoint & 0x3F)
        return bytes([byte1, byte2]).decode("utf-8")
    elif codepoint <= 0xFFFF:
        # 3-byte UTF-8
        byte1 = 0xE0 | (codepoint >> 12)
        byte2 = 0x80 | ((codepoint >> 6) & 0x3F)
        byte3 = 0x80 | (codepoint & 0x3F)
        return bytes([byte1, byte2, byte3]).decode("utf-8")
    elif codepoint <= 0x10FFFF:
        # 4-byte UTF-8
        byte1 = 0xF0 | (codepoint >> 18)
        byte2 = 0x80 | ((codepoint >> 12) & 0x3F)
        byte3 = 0x80 | ((codepoint >> 6) & 0x3F)
        byte4 = 0x80 | (codepoint & 0x3F)
        return bytes([byte1, byte2, byte3, byte4]).decode("utf-8")
    else:
        raise ValueError(f"Invalid codepoint: {codepoint} (must be <= 0x10FFFF)")


def print_image(image: "Image.Image", flags: int) -> list[str]:
    """Print an image as ASCII art using block characters and ANSI colors.

    This function is ported from the C++ TerminalImageViewer's printImage function.
    It processes the image in 4x8 pixel blocks, generating character data using either
    createCharData or findCharData based on flags, and produces ANSI color codes and
    Unicode block characters.

    Args:
        image: PIL/Pillow Image object to convert to ASCII art
        flags: Bit flags controlling rendering options:
               - FLAG_NOOPT: Use simple lower half block character (0x2584)
               - FLAG_MODE_256: Use 256-color mode instead of 24-bit true color
               - FLAG_BG/FLAG_FG: Used internally for color output

    Returns:
        String containing ANSI color codes and Unicode characters representing the image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size

    # Create get_pixel function for PIL Image
    def get_pixel(x: int, y: int) -> int:
        """Get RGB pixel value at coordinates (x, y).

        Returns RGB value as integer in format 0xRRGGBB.
        Returns black (0x000000) for coordinates outside image bounds.
        """
        if x < 0 or y < 0 or x >= width or y >= height:
            return 0x000000  # Black for out-of-bounds pixels

        pixel = image.getpixel((x, y))
        r, g, b = pixel if isinstance(pixel, tuple) and len(pixel) >= 3 else (0, 0, 0)
        return (r << 16) | (g << 8) | b

    result = []
    last_char_data = CharData()

    # Process image in 4x8 blocks (matching C++ implementation)
    for y in range(0, height - 7, 8):  # height - 7 ensures we have at least 8 pixels
        line_output = []

        for x in range(0, width - 3, 4):  # width - 3 ensures we have at least 4 pixels
            # Choose character data generation method based on flags
            if flags & FLAG_NOOPT:
                # Simple mode: use lower half block with fixed pattern
                char_data = create_char_data(get_pixel, x, y, 0x2584, 0x0000FFFF)
            else:
                # Optimized mode: find best matching character
                char_data = find_char_data(get_pixel, x, y, flags)

            # Output background color if it changed or this is the first character in the line
            if x == 0 or char_data.bg_color != last_char_data.bg_color:
                bg_color_code = print_term_color(
                    flags | FLAG_BG,
                    char_data.bg_color[0],
                    char_data.bg_color[1],
                    char_data.bg_color[2],
                )
                line_output.append(bg_color_code)

            # Output foreground color if it changed or this is the first character in the line
            if x == 0 or char_data.fg_color != last_char_data.fg_color:
                fg_color_code = print_term_color(
                    flags | FLAG_FG,
                    char_data.fg_color[0],
                    char_data.fg_color[1],
                    char_data.fg_color[2],
                )
                line_output.append(fg_color_code)

            # Output the Unicode character
            char_str = print_codepoint(char_data.codepoint)
            line_output.append(char_str)

            last_char_data = char_data

        line_output.append("\x1b[0m")
        result.append("".join(line_output))

    return result


def find_char_data(
    get_pixel: GetPixelFunction, x0: int, y0: int, flags: int
) -> CharData:
    """Find the best character and colors for a 4x8 pixel area.

    This function analyzes a 4x8 pixel area to find the best matching character
    from the BITMAPS array and determines the optimal colors. It has two modes:

    1. Direct mode: When the 2 most common colors represent >50% of pixels,
       it creates a bitmap based on distance to these colors.
    2. Color channel split mode: When colors are more distributed, it splits
       on the color channel with the greatest range.

    The function then searches through all available bitmap patterns (including
    inverted versions) to find the best match by counting differing bits.

    Args:
        get_pixel: Function that returns RGB color for pixel at (x, y)
        x0: Starting x coordinate of the 4x8 pixel area
        y0: Starting y coordinate of the 4x8 pixel area
        flags: Flags controlling which characters are allowed

    Returns:
        CharData with the best character and colors for this area
    """
    # Initialize min/max values for each color channel
    min_vals = [255, 255, 255]
    max_vals = [0, 0, 0]
    count_per_color: Dict[int, int] = {}

    # Analyze the 4x8 pixel area to determine color distribution and ranges
    for y in range(8):
        for x in range(4):
            rgb = get_pixel(x0 + x, y0 + y)
            color = 0

            for i in range(3):
                channel_val = get_channel(rgb, i)
                min_vals[i] = min(min_vals[i], channel_val)
                max_vals[i] = max(max_vals[i], channel_val)
                color = (color << 8) | channel_val

            count_per_color[color] = count_per_color.get(color, 0) + 1

    sorted_colors = sorted(count_per_color.items(), key=lambda x: x[1], reverse=True)

    max_count_color_1 = sorted_colors[0][0]
    count2 = sorted_colors[0][1]
    max_count_color_2 = max_count_color_1

    if len(sorted_colors) > 1:
        count2 += sorted_colors[1][1]
        max_count_color_2 = sorted_colors[1][0]

    bits = 0
    direct = count2 > (8 * 4) // 2

    if direct:
        # Direct mode: create bitmap based on distance to two most common colors
        for y in range(8):
            for x in range(4):
                bits = bits << 1
                rgb = get_pixel(x0 + x, y0 + y)

                d1 = 0
                d2 = 0

                for i in range(3):
                    shift = 16 - 8 * i
                    c1 = (max_count_color_1 >> shift) & 255
                    c2 = (max_count_color_2 >> shift) & 255
                    c = get_channel(rgb, i)
                    d1 += (c1 - c) * (c1 - c)
                    d2 += (c2 - c) * (c2 - c)

                if d1 > d2:
                    bits |= 1
    else:
        # Color channel split mode: find channel with greatest range
        split_index = 0
        best_split = 0

        for i in range(3):
            range_val = max_vals[i] - min_vals[i]
            if range_val > best_split:
                best_split = range_val
                split_index = i

        split_value = min_vals[split_index] + best_split // 2

        for y in range(8):
            for x in range(4):
                bits = bits << 1
                if get_channel(get_pixel(x0 + x, y0 + y), split_index) > split_value:
                    bits |= 1

    best_diff = 8
    best_pattern = 0x0000FFFF
    codepoint = 0x2584
    inverted = False

    i = 0
    while i < len(BITMAPS) and BITMAPS[i + 1] != END_MARKER:
        if (BITMAPS[i + 2] & flags) != BITMAPS[i + 2]:
            i += 3
            continue

        pattern = BITMAPS[i]

        for j in range(2):
            diff = bin(pattern ^ bits).count("1")

            if diff < best_diff:
                best_pattern = BITMAPS[i]
                codepoint = BITMAPS[i + 1]
                best_diff = diff
                inverted = best_pattern != pattern

            pattern = ~pattern & 0xFFFFFFFF

        i += 3

    if direct:
        result = CharData()
        result.codepoint = codepoint

        if inverted:
            max_count_color_1, max_count_color_2 = max_count_color_2, max_count_color_1

        for i in range(3):
            shift = 16 - 8 * i
            result.fg_color[i] = (max_count_color_2 >> shift) & 255
            result.bg_color[i] = (max_count_color_1 >> shift) & 255

        return result
    else:
        return create_char_data(get_pixel, x0, y0, codepoint, best_pattern)


