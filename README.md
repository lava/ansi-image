# ansi-image

A pure Python library for displaying images in the terminal using ANSI color codes. This is a straight port of the algorithm from the [tiv (TerminalImageViewer)](https://github.com/stefanhaustein/TerminalImageViewer) command, making it available as a Python library.

## Installation

```bash
pip install ansi-image
```

## Usage

### Basic Usage

```python
from PIL import Image
from ansi_image import AnsiImage

# Load an image
img = Image.open("path/to/your/image.jpg")

# Create an AnsiImage object (keeps full image in memory)
ansi_img = AnsiImage(img)

# Print to terminal
print(ansi_img.render())
```

### Memory Efficient Usage

The main `AnsiImage` class keeps a full copy of the source image in memory. For memory efficiency, you can use the `render()` method to get only the text version:

```python
from PIL import Image
from ansi_image import AnsiImage

# Load and render directly to text (doesn't keep image in memory)
rendered = AnsiImage.from_image_file("path/to/your/image.jpg")
print(rendered)

# Or from an existing PIL Image
img = Image.open("path/to/your/image.jpg")
rendered = AnsiImage.from_image(img)
print(rendered)
```

### Size Control

```python
# Specify dimensions (terminal columns x rows)
rendered = ansi_img.render(output_width=80, output_height=24)
print(rendered)

# Auto-fit to terminal size (default)
rendered = ansi_img.render()
print(rendered)

# Using format strings
print(f"{ansi_img:w=40,h=20}")
print(f"{ansi_img:width=60}")
```

### Background Fill

```python
# Add background color to fill the entire bounding box
rendered = ansi_img.render(fill="#ffffff")  # White background
print(rendered)

# Using format strings
print(f"{ansi_img:w=40,bg=#000000}")  # Black background
```

### Command Line Tool

The package also includes a command-line tool:

```bash
print-image path/to/your/image.jpg
print-image --width 80 --height 24 image.jpg
print-image --fill "#ffffff" image.jpg
```

## API Reference

### AnsiImage

Main class that stores the original PIL Image and provides rendering methods.

- `AnsiImage(image)` - Create from PIL Image object
- `render(output_width=None, output_height=None, flags=0, fill=None)` - Render to RenderedAnsiImage
- `AnsiImage.from_image(img, ...)` - Static method to render directly from PIL Image
- `AnsiImage.from_image_file(path, ...)` - Static method to load and render from file

### RenderedAnsiImage

Contains the pre-rendered text representation that can be printed.

- `str(rendered)` - Convert to printable string
- `rendered.width` - Width in terminal columns
- `rendered.height` - Height in terminal rows
- `rendered.data` - List of strings with ANSI codes

## Memory Usage

- `AnsiImage` objects keep the full PIL Image in memory, allowing multiple renders with different parameters
- `RenderedAnsiImage` objects only contain the text representation
- For one-time rendering, use the static methods `from_image()` or `from_image_file()` to avoid keeping the image in memory

## Algorithm

The rendering algorithm is a direct port from the C++ implementation in [TerminalImageViewer](https://github.com/stefanhaustein/TerminalImageViewer), providing the same high-quality terminal image display in pure Python.