#!/usr/bin/env python3
"""Command-line tool for printing images to the terminal using ANSI colors."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ansi_image.ansi_image import AnsiImage


def main() -> None:
    """Main entry point for the print-image command-line tool."""
    parser = argparse.ArgumentParser(
        description="Print an image to the terminal using ANSI colors",
        prog="print-image",
        add_help=False
    )
    
    parser.add_argument(
        "filename",
        help="Path to the image file to display"
    )
    
    parser.add_argument(
        "-w", "--width",
        type=int,
        help="Maximum width in terminal columns (if height not specified, height will be calculated to preserve aspect ratio)"
    )
    
    parser.add_argument(
        "-h", "--height",
        type=int,
        help="Maximum height in terminal rows (if width not specified, width will be calculated to preserve aspect ratio)"
    )
    
    parser.add_argument(
        "--flags",
        type=int,
        default=0,
        help="Rendering flags (0=default, 1=simple mode with FLAG_NOOPT)"
    )
    
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Fill background to fit the entire bounding box"
    )
    
    parser.add_argument(
        "--bg-color",
        default="#000000",
        help="Background color as hex string when using --fill (default: #000000)"
    )
    
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit"
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.filename)
    if not image_path.exists():
        print(f"Error: File '{args.filename}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not image_path.is_file():
        print(f"Error: '{args.filename}' is not a file.", file=sys.stderr)
        sys.exit(1)
    
    width: Optional[int] = getattr(args, 'width', None)
    height: Optional[int] = getattr(args, 'height', None)
    
    try:
        fill_color = getattr(args, 'bg_color', '#000000') if args.fill else None
        ansi_img = AnsiImage.from_file(str(image_path), width, height, args.flags, fill_color)
        print(ansi_img)
    
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()