"""Tables and constants for ASCII art generation.

This module contains the bitmap patterns, color tables, and constants
used for ASCII art character generation.
"""

# Constants
FLAG_FG = 1
FLAG_BG = 2
FLAG_MODE_256 = 4
FLAG_24BIT = 8
FLAG_NOOPT = 16
FLAG_TELETEXT = 32

# Color saturation value steps from 0 to 255
COLOR_STEP_COUNT = 6
COLOR_STEPS = [0, 0x5F, 0x87, 0xAF, 0xD7, 0xFF]

# Grayscale saturation value steps from 0 to 255
GRAYSCALE_STEP_COUNT = 24
GRAYSCALE_STEPS = [
    0x08,
    0x12,
    0x1C,
    0x26,
    0x30,
    0x3A,
    0x44,
    0x4E,
    0x58,
    0x62,
    0x6C,
    0x76,
    0x80,
    0x8A,
    0x94,
    0x9E,
    0xA8,
    0xB2,
    0xBC,
    0xC6,
    0xD0,
    0xDA,
    0xE4,
    0xEE,
]

END_MARKER = 0

# An interleaved map of 4x8 bit character bitmaps (each hex digit represents a row)
# to the corresponding unicode code point.
# Each entry is a tuple of (bitmap, codepoint, flags)
BITMAPS = [
    0x00000000,
    0x00A0,
    0,
    # Block graphics
    # 0xffff0000, 0x2580, 0,  # upper 1/2; redundant with inverse lower 1/2
    0x0000000F,
    0x2581,
    0,  # lower 1/8
    0x000000FF,
    0x2582,
    0,  # lower 1/4
    0x00000FFF,
    0x2583,
    0,
    0x0000FFFF,
    0x2584,
    0,  # lower 1/2
    0x000FFFFF,
    0x2585,
    0,
    0x00FFFFFF,
    0x2586,
    0,  # lower 3/4
    0x0FFFFFFF,
    0x2587,
    0,
    # 0xffffffff, 0x2588,  # full; redundant with inverse space
    0xEEEEEEEE,
    0x258A,
    0,  # left 3/4
    0xCCCCCCCC,
    0x258C,
    0,  # left 1/2
    0x88888888,
    0x258E,
    0,  # left 1/4
    0x0000CCCC,
    0x2596,
    0,  # quadrant lower left
    0x00003333,
    0x2597,
    0,  # quadrant lower right
    0xCCCC0000,
    0x2598,
    0,  # quadrant upper left
    # 0xccccffff, 0x2599,  # 3/4 redundant with inverse 1/4
    0xCCCC3333,
    0x259A,
    0,  # diagonal 1/2
    # 0xffffcccc, 0x259b,  # 3/4 redundant
    # 0xffff3333, 0x259c,  # 3/4 redundant
    0x33330000,
    0x259D,
    0,  # quadrant upper right
    # 0x3333cccc, 0x259e,  # 3/4 redundant
    # 0x3333ffff, 0x259f,  # 3/4 redundant
    # Line drawing subset: no double lines, no complex light lines
    0x000FF000,
    0x2501,
    0,  # Heavy horizontal
    0x66666666,
    0x2503,
    0,  # Heavy vertical
    0x00077666,
    0x250F,
    0,  # Heavy down and right
    0x000EE666,
    0x2513,
    0,  # Heavy down and left
    0x66677000,
    0x2517,
    0,  # Heavy up and right
    0x666EE000,
    0x251B,
    0,  # Heavy up and left
    0x66677666,
    0x2523,
    0,  # Heavy vertical and right
    0x666EE666,
    0x252B,
    0,  # Heavy vertical and left
    0x000FF666,
    0x2533,
    0,  # Heavy down and horizontal
    0x666FF000,
    0x253B,
    0,  # Heavy up and horizontal
    0x666FF666,
    0x254B,
    0,  # Heavy cross
    0x000CC000,
    0x2578,
    0,  # Bold horizontal left
    0x00066000,
    0x2579,
    0,  # Bold horizontal up
    0x00033000,
    0x257A,
    0,  # Bold horizontal right
    0x00066000,
    0x257B,
    0,  # Bold horizontal down
    0x06600660,
    0x254F,
    0,  # Heavy double dash vertical
    0x000F0000,
    0x2500,
    0,  # Light horizontal
    0x0000F000,
    0x2500,
    0,  #
    0x44444444,
    0x2502,
    0,  # Light vertical
    0x22222222,
    0x2502,
    0,
    0x000E0000,
    0x2574,
    0,  # light left
    0x0000E000,
    0x2574,
    0,  # light left
    0x44440000,
    0x2575,
    0,  # light up
    0x22220000,
    0x2575,
    0,  # light up
    0x00030000,
    0x2576,
    0,  # light right
    0x00003000,
    0x2576,
    0,  # light right
    0x00004444,
    0x2577,
    0,  # light down
    0x00002222,
    0x2577,
    0,  # light down
    # Misc technical
    0x44444444,
    0x23A2,
    0,  # [ extension
    0x22222222,
    0x23A5,
    0,  # ] extension
    0x0F000000,
    0x23BA,
    0,  # Horizontal scanline 1
    0x00F00000,
    0x23BB,
    0,  # Horizontal scanline 3
    0x00000F00,
    0x23BC,
    0,  # Horizontal scanline 7
    0x000000F0,
    0x23BD,
    0,  # Horizontal scanline 9
    # Geometrical shapes. Tricky because some of them are too wide.
    # 0x00ffff00, 0x25fe, 0,  # Black medium small square
    0x00066000,
    0x25AA,
    0,  # Black small square
    # 0x11224488, 0x2571, 0,  # diagonals
    # 0x88442211, 0x2572, 0,
    # 0x99666699, 0x2573, 0,
    # 0x000137f0, 0x25e2, 0,  # Triangles
    # 0x0008cef0, 0x25e3, 0,
    # 0x000fec80, 0x25e4, 0,
    # 0x000f7310, 0x25e5, 0,
    # Teletext / legacy graphics 3x2 block character codes.
    # Using a 3-2-3 pattern consistently, perhaps we should create automatic
    # variations....
    0xCCC00000,
    0xFB00,
    FLAG_TELETEXT,
    0x33300000,
    0xFB01,
    FLAG_TELETEXT,
    0xFFF00000,
    0xFB02,
    FLAG_TELETEXT,
    0x000CC000,
    0xFB03,
    FLAG_TELETEXT,
    0xCCCCC000,
    0xFB04,
    FLAG_TELETEXT,
    0x333CC000,
    0xFB05,
    FLAG_TELETEXT,
    0xFFFCC000,
    0xFB06,
    FLAG_TELETEXT,
    0x00033000,
    0xFB07,
    FLAG_TELETEXT,
    0xCCC33000,
    0xFB08,
    FLAG_TELETEXT,
    0x33333000,
    0xFB09,
    FLAG_TELETEXT,
    0xFFF33000,
    0xFB0A,
    FLAG_TELETEXT,
    0x000FF000,
    0xFB0B,
    FLAG_TELETEXT,
    0xCCCFF000,
    0xFB0C,
    FLAG_TELETEXT,
    0x333FF000,
    0xFB0D,
    FLAG_TELETEXT,
    0xFFFFF000,
    0xFB0E,
    FLAG_TELETEXT,
    0x00000CCC,
    0xFB0F,
    FLAG_TELETEXT,
    0xCCC00CCC,
    0xFB10,
    FLAG_TELETEXT,
    0x33300CCC,
    0xFB11,
    FLAG_TELETEXT,
    0xFFF00CCC,
    0xFB12,
    FLAG_TELETEXT,
    0x000CCCCC,
    0xFB13,
    FLAG_TELETEXT,
    0x333CCCCC,
    0xFB14,
    FLAG_TELETEXT,
    0xFFFCCCCC,
    0xFB15,
    FLAG_TELETEXT,
    0x00033CCC,
    0xFB16,
    FLAG_TELETEXT,
    0xCCC33CCC,
    0xFB17,
    FLAG_TELETEXT,
    0x33333CCC,
    0xFB18,
    FLAG_TELETEXT,
    0xFFF33CCC,
    0xFB19,
    FLAG_TELETEXT,
    0x000FFCCC,
    0xFB1A,
    FLAG_TELETEXT,
    0xCCCFFCCC,
    0xFB1B,
    FLAG_TELETEXT,
    0x333FFCCC,
    0xFB1C,
    FLAG_TELETEXT,
    0xFFFFFCCC,
    0xFB1D,
    FLAG_TELETEXT,
    0x00000333,
    0xFB1E,
    FLAG_TELETEXT,
    0xCCC00333,
    0xFB1F,
    FLAG_TELETEXT,
    0x33300333,
    0x1B20,
    FLAG_TELETEXT,
    0xFFF00333,
    0x1B21,
    FLAG_TELETEXT,
    0x000CC333,
    0x1B22,
    FLAG_TELETEXT,
    0xCCCCC333,
    0x1B23,
    FLAG_TELETEXT,
    0x333CC333,
    0x1B24,
    FLAG_TELETEXT,
    0xFFFCC333,
    0x1B25,
    FLAG_TELETEXT,
    0x00033333,
    0x1B26,
    FLAG_TELETEXT,
    0xCCC33333,
    0x1B27,
    FLAG_TELETEXT,
    0xFFF33333,
    0x1B28,
    FLAG_TELETEXT,
    0x000FF333,
    0x1B29,
    FLAG_TELETEXT,
    0xCCCFF333,
    0x1B2A,
    FLAG_TELETEXT,
    0x333FF333,
    0x1B2B,
    FLAG_TELETEXT,
    0xFFFFF333,
    0x1B2C,
    FLAG_TELETEXT,
    0x00000FFF,
    0x1B2D,
    FLAG_TELETEXT,
    0xCCC00FFF,
    0x1B2E,
    FLAG_TELETEXT,
    0x33300FFF,
    0x1B2F,
    FLAG_TELETEXT,
    0xFFF00FFF,
    0x1B30,
    FLAG_TELETEXT,
    0x000CCFFF,
    0x1B31,
    FLAG_TELETEXT,
    0xCCCCCFFF,
    0x1B32,
    FLAG_TELETEXT,
    0x333CCFFF,
    0x1B33,
    FLAG_TELETEXT,
    0xFFFCCFFF,
    0x1B34,
    FLAG_TELETEXT,
    0x00033FFF,
    0x1B35,
    FLAG_TELETEXT,
    0xCCC33FFF,
    0x1B36,
    FLAG_TELETEXT,
    0x33333FFF,
    0x1B37,
    FLAG_TELETEXT,
    0xFFF33FFF,
    0x1B38,
    FLAG_TELETEXT,
    0x000FFFFF,
    0x1B39,
    FLAG_TELETEXT,
    0xCCCFFFFF,
    0x1B3A,
    FLAG_TELETEXT,
    0x333FFFFF,
    0x1B3B,
    FLAG_TELETEXT,
    0,
    END_MARKER,
    0,  # End marker
]