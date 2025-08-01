"""ANSI Image class for storing and displaying terminal-based images."""

import os
from typing import List, Optional, Tuple
from .ascii_art import to_ascii
from PIL import Image


def _get_terminal_dimensions(
    output_width: Optional[int], output_height: Optional[int]
) -> Tuple[int, int]:
    """Get terminal dimensions, validating that either both or neither are provided.
    
    Args:
        output_width: Optional width in terminal columns
        output_height: Optional height in terminal rows
        
    Returns:
        Tuple of (width, height) as integers
        
    Raises:
        ValueError: If only one of output_width or output_height is provided
    """
    # Validate that either both or neither dimensions are provided
    if (output_width is None) != (output_height is None):
        raise ValueError("Either both output_width and output_height must be provided, or both must be None")
    
    # If dimensions not provided, get terminal size
    if output_width is None:
        try:
            terminal_size = os.get_terminal_size()
            output_width = terminal_size.columns
            output_height = terminal_size.lines
        except OSError:
            # Fallback to default size if terminal size cannot be determined
            output_width = 80
            output_height = 24
    
    # At this point, both output_width and output_height are guaranteed to be integers
    assert output_width is not None and output_height is not None
    return output_width, output_height


class AnsiImage:
    """A class to represent an ANSI-colored image for terminal display.
    
    This class stores image data as a collection of ANSI-colored strings that can
    be printed to the terminal to display the image.
    
    Attributes:
        width: The width of the image in terminal character columns
        height: The height of the image in terminal character rows  
        data: List of strings containing ANSI color codes and characters
    """
    
    def __init__(self, width: int, height: int, data: List[str]) -> None:
        """Initialize an AnsiImage.
        
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
        """Return a string representation of the AnsiImage object.
        
        Returns:
            A string showing the object's type and dimensions
        """
        return f"AnsiImage(width={self.width}, height={self.height}, lines={repr(self.data)})"
    
    @staticmethod
    def from_image(
        img: "Image.Image", 
        output_width: Optional[int] = None, 
        output_height: Optional[int] = None, 
        flags: int = 0
    ) -> "AnsiImage":
        """Create an AnsiImage from a PIL Image.
        
        This is a convenience method that calls the to_ascii function from the
        ascii_art module to convert a PIL Image to an AnsiImage.
        
        Args:
            img: PIL/Pillow Image object to convert to ASCII art
            output_width: Maximum width for the output (in terminal character columns).
                         If None, uses current terminal width. Must be None if output_height is None.
            output_height: Maximum height for the output (in terminal character rows).
                          If None, uses current terminal height. Must be None if output_width is None.
            flags: Bit flags controlling rendering options
            
        Returns:
            An AnsiImage object containing the converted image
            
        Raises:
            ValueError: If only one of output_width or output_height is provided
        """
        output_width, output_height = _get_terminal_dimensions(output_width, output_height)
        return to_ascii(img, output_width, output_height, flags)
    
    @staticmethod
    def from_image_file(
        file_path: str, 
        output_width: Optional[int] = None, 
        output_height: Optional[int] = None, 
        flags: int = 0
    ) -> "AnsiImage":
        """Create an AnsiImage from an image file.
        
        This is a convenience method that loads an image from a file path
        and then converts it to an AnsiImage using the from_image method.
        
        Args:
            file_path: Path to the image file to load
            output_width: Maximum width for the output (in terminal character columns).
                         If None, uses current terminal width. Must be None if output_height is None.
            output_height: Maximum height for the output (in terminal character rows).
                          If None, uses current terminal height. Must be None if output_width is None.
            flags: Bit flags controlling rendering options
            
        Returns:
            An AnsiImage object containing the converted image
            
        Raises:
            FileNotFoundError: If the image file does not exist
            IOError: If the image file cannot be opened or is not a valid image
            ValueError: If only one of output_width or output_height is provided
        """
        img = Image.open(file_path)
        return AnsiImage.from_image(img, output_width, output_height, flags)