"""ANSI Image class for storing and displaying terminal-based images."""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


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
        img: "Image.Image", output_width: int, output_height: int, flags: int = 0
    ) -> "AnsiImage":
        """Create an AnsiImage from a PIL Image.
        
        This is a convenience method that calls the to_ascii function from the
        ascii_art module to convert a PIL Image to an AnsiImage.
        
        Args:
            img: PIL/Pillow Image object to convert to ASCII art
            output_width: Maximum width for the output (in terminal character columns)
            output_height: Maximum height for the output (in terminal character rows)
            flags: Bit flags controlling rendering options
            
        Returns:
            An AnsiImage object containing the converted image
        """
        from .ascii_art import to_ascii
        return to_ascii(img, output_width, output_height, flags)