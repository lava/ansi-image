"""ANSI Image class for storing and displaying terminal-based images."""

import os
import re
from typing import List, Optional, Tuple
from PIL import Image


def _get_terminal_dimensions(
    output_width: Optional[int], output_height: Optional[int], img: Optional["Image.Image"] = None
) -> Tuple[int, int]:
    """Get terminal dimensions, calculating missing dimension to preserve aspect ratio if needed.
    
    Args:
        output_width: Optional width in terminal columns
        output_height: Optional height in terminal rows  
        img: Optional PIL Image to calculate aspect ratio from
        
    Returns:
        Tuple of (width, height) as integers
        
    Raises:
        ValueError: If both dimensions are None but no image is provided for aspect ratio calculation
    """
    # If both dimensions are provided, use them as-is
    if output_width is not None and output_height is not None:
        return output_width, output_height
    
    # If neither dimension is provided, get terminal size
    if output_width is None and output_height is None:
        try:
            terminal_size = os.get_terminal_size()
            output_width = terminal_size.columns
            output_height = terminal_size.lines
        except OSError:
            # Fallback to default size if terminal size cannot be determined
            output_width = 80
            output_height = 24
        return output_width, output_height
    
    # If only one dimension is provided, calculate the other based on aspect ratio
    if img is None:
        raise ValueError("Image must be provided to calculate missing dimension based on aspect ratio")
    
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height
    
    if output_width is not None:
        # Calculate height from width while preserving aspect ratio
        # Account for character cell ratio (4x8 pixels per cell)
        output_height = int((output_width * 4) / (aspect_ratio * 8))
    elif output_height is not None:
        # Calculate width from height while preserving aspect ratio  
        # Account for character cell ratio (4x8 pixels per cell)
        output_width = int((output_height * 8 * aspect_ratio) / 4)
    
    # At this point, both dimensions should be set
    assert output_width is not None and output_height is not None
    return output_width, output_height


class RenderedAnsiImage:
    """A class to represent a rendered ANSI-colored image for terminal display.
    
    This class stores pre-rendered image data as a collection of ANSI-colored strings that can
    be printed to the terminal to display the image.
    
    Attributes:
        width: The width of the image in terminal character columns
        height: The height of the image in terminal character rows  
        data: List of strings containing ANSI color codes and characters
    """
    
    def __init__(self, width: int, height: int, data: List[str]) -> None:
        """Initialize a RenderedAnsiImage.
        
        Args:
            width: Width of the image in terminal character columns
            height: Height of the image in terminal character rows
            data: List of strings containing the image data with ANSI codes
        """
        self.width = width
        self.height = height
        self.data = data
    
    def __str__(self) -> str:
        """Convert the image to a string for printing.
        
        Returns:
            A string representation of the image that can be printed to display it
        """
        return "\n".join(self.data)
    
    def __repr__(self) -> str:
        """Return a string representation of the RenderedAnsiImage object.
        
        Returns:
            A string showing the object's type and dimensions
        """
        return f"RenderedAnsiImage(width={self.width}, height={self.height}, lines={repr(self.data)})"


class AnsiImage:
    """A class to store image data and render it to terminal display.
    
    This class stores the original PIL Image and provides methods to render
    it to ANSI-colored terminal output with various options.
    
    Attributes:
        image: The PIL Image object containing the image data
    """
    
    def __init__(self, image: "Image.Image") -> None:
        """Initialize an AnsiImage with PIL Image data.
        
        Args:
            image: PIL/Pillow Image object to store
        """
        self.image = image
    
    def render(
        self,
        output_width: Optional[int] = None, 
        output_height: Optional[int] = None, 
        flags: int = 0,
        fill: Optional[str] = None
    ) -> "RenderedAnsiImage":
        """Render the image to ANSI terminal output.
        
        Args:
            output_width: Maximum width for the output (in terminal character columns).
                         If None, uses current terminal width or calculates from height and aspect ratio.
            output_height: Maximum height for the output (in terminal character rows).
                          If None, uses current terminal height or calculates from width and aspect ratio.
            flags: Bit flags controlling rendering options
            fill: Optional background color as hex string (e.g., "#ffffff" for white).
                 If provided, adds background pixels so the image fills the whole bounding box.
            
        Returns:
            A RenderedAnsiImage object containing the converted image
        """
        output_width, output_height = _get_terminal_dimensions(output_width, output_height, self.image)
        return to_ascii(self.image, output_width, output_height, flags, fill)
    
    def __format__(self, format_spec: str) -> str:
        """Format the AnsiImage with custom format specifiers.
        
        Supports format specifiers like:
        - w=10 or width=10: Set width to 10 columns
        - h=20 or height=20: Set height to 20 rows  
        - bg=#ffffff or bg=ffffff: Set background color
        - flags=1: Set rendering flags
        
        Multiple specifiers can be combined with commas:
        f"{image:w=10,h=20,bg=#ffffff}"
        
        Args:
            format_spec: Format specification string
            
        Returns:
            Formatted string representation of the rendered image
        """
        # Default values
        width: Optional[int] = None
        height: Optional[int] = None
        fill: Optional[str] = None
        flags: int = 0
        
        # Parse format specifiers
        if format_spec:
            # Split by commas and process each specifier
            for spec in format_spec.split(','):
                spec = spec.strip()
                if '=' not in spec:
                    continue
                    
                key, value = spec.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in ('w', 'width'):
                    try:
                        width = int(value)
                    except ValueError:
                        raise ValueError(f"Invalid width value: {value}")
                elif key in ('h', 'height'):
                    try:
                        height = int(value)
                    except ValueError:
                        raise ValueError(f"Invalid height value: {value}")
                elif key in ('bg', 'background', 'fill'):
                    # Accept both #ffffff and ffffff formats
                    if not value.startswith('#') and re.match(r'^[0-9a-fA-F]{6}$', value):
                        value = '#' + value
                    fill = value
                elif key == 'flags':
                    try:
                        flags = int(value)
                    except ValueError:
                        raise ValueError(f"Invalid flags value: {value}")
                else:
                    raise ValueError(f"Unknown format specifier: {key}")
        
        # Render the image with the specified parameters
        rendered = self.render(output_width=width, output_height=height, flags=flags, fill=fill)
        return str(rendered)
    
    @staticmethod
    def from_image(
        img: "Image.Image", 
        output_width: Optional[int] = None, 
        output_height: Optional[int] = None, 
        flags: int = 0,
        fill: Optional[str] = None
    ) -> "RenderedAnsiImage":
        """Create a RenderedAnsiImage from a PIL Image.
        
        This is a convenience method that calls the to_ascii function from the
        algorithms module to convert a PIL Image to a RenderedAnsiImage.
        
        Args:
            img: PIL/Pillow Image object to convert to ASCII art
            output_width: Maximum width for the output (in terminal character columns).
                         If None, uses current terminal width or calculates from height and aspect ratio.
            output_height: Maximum height for the output (in terminal character rows).
                          If None, uses current terminal height or calculates from width and aspect ratio.
            flags: Bit flags controlling rendering options
            fill: Optional background color as hex string (e.g., "#ffffff" for white).
                 If provided, adds background pixels so the image fills the whole bounding box.
            
        Returns:
            A RenderedAnsiImage object containing the converted image
        """
        output_width, output_height = _get_terminal_dimensions(output_width, output_height, img)
        return to_ascii(img, output_width, output_height, flags, fill)
    
    @staticmethod
    def from_image_file(
        file_path: str, 
        output_width: Optional[int] = None, 
        output_height: Optional[int] = None, 
        flags: int = 0,
        fill: Optional[str] = None
    ) -> "RenderedAnsiImage":
        """Create a RenderedAnsiImage from an image file.
        
        This is a convenience method that loads an image from a file path
        and then converts it to a RenderedAnsiImage using the from_image method.
        
        Args:
            file_path: Path to the image file to load
            output_width: Maximum width for the output (in terminal character columns).
                         If None, uses current terminal width or calculates from height and aspect ratio.
            output_height: Maximum height for the output (in terminal character rows).
                          If None, uses current terminal height or calculates from width and aspect ratio.
            flags: Bit flags controlling rendering options
            fill: Optional background color as hex string (e.g., "#ffffff" for white).
                 If provided, adds background pixels so the image fills the whole bounding box.
            
        Returns:
            A RenderedAnsiImage object containing the converted image
            
        Raises:
            FileNotFoundError: If the image file does not exist
            IOError: If the image file cannot be opened or is not a valid image
        """
        img = Image.open(file_path)
        return AnsiImage.from_image(img, output_width, output_height, flags, fill)


def to_ascii(
    img: "Image.Image", output_width: int, output_height: int, flags: int = 0, fill: Optional[str] = None
) -> "RenderedAnsiImage":
    """Convert an image to ASCII art with resizing logic from TerminalImageViewer.

    This function implements the same resize logic as tiv.cpp:360-366, scaling the
    image down to fit within the specified dimensions while maintaining aspect ratio.

    Args:
        img: PIL/Pillow Image object to convert to ASCII art
        output_width: Maximum width for the output (in terminal character columns)
        output_height: Maximum height for the output (in terminal character rows)
        flags: Bit flags controlling rendering options (same as print_image)
        fill: Optional background color as hex string (e.g., "#ffffff" for white).
             If provided, adds background pixels so the image fills the whole bounding box.

    Returns:
        List of strings containing ANSI color codes and Unicode characters representing the image
    """
    from ansi_image.algorithms import print_image

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Get original image dimensions
    original_width, original_height = img.size

    # Convert terminal dimensions to pixel dimensions
    # Each character cell represents 4x8 pixels in the ASCII art
    max_pixel_width = output_width * 4
    max_pixel_height = output_height * 8

    # Apply resize logic from tiv.cpp:360-366
    if original_width > max_pixel_width or original_height > max_pixel_height:
        # Calculate scale factor that fits image within target dimensions
        # This matches the fitted_within logic: min(container.width/width, container.height/height)
        scale = min(
            max_pixel_width / original_width, max_pixel_height / original_height
        )

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image using high-quality resampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        new_width, new_height = original_width, original_height

    # Apply fill background if specified
    if fill is not None:
        # Parse hex color string (supports both #ffffff and ffffff formats)
        fill_color_str = fill.lstrip('#')
        if len(fill_color_str) != 6:
            raise ValueError(f"Invalid fill color '{fill}': expected 6-character hex string")
        
        try:
            fill_r = int(fill_color_str[0:2], 16)
            fill_g = int(fill_color_str[2:4], 16)
            fill_b = int(fill_color_str[4:6], 16)
        except ValueError:
            raise ValueError(f"Invalid fill color '{fill}': not a valid hex color")
        
        # Create a new image with the target dimensions and fill color
        filled_img = Image.new("RGB", (max_pixel_width, max_pixel_height), (fill_r, fill_g, fill_b))
        
        # Center the original image on the filled background
        x_offset = (max_pixel_width - new_width) // 2
        y_offset = (max_pixel_height - new_height) // 2
        filled_img.paste(img, (x_offset, y_offset))
        
        img = filled_img
        new_width, new_height = max_pixel_width, max_pixel_height

    # Convert the (possibly resized and filled) image to ASCII using print_image
    lines = print_image(img, flags)
    return RenderedAnsiImage(width=new_width // 4, height=new_height // 8, data=lines)