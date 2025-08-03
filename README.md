# ansi-image

A library and tool for displaying images in the terminal, using ANSI color codes.

The pixel-to-ansi algorithm is a straight port from the [tiv (TerminalImageViewer)](https://github.com/stefanhaustein/TerminalImageViewer) implementation, so this library should
produce exactly the same images.

The main difference compared to most other terminal image viewers is that this
project is mainly designed as a library, instead of a command-line application.

For *displaying* the images, your terminal must support ANSI, 24 bit true color
and unicode rendering. Unless you're a retro computing enthusiast, your current
terminal does support these.

## Usage

### Basic Usage

```python
from ansi_image import AnsiImage

image = AnsiImage.from_file("tests/test_image.png")
print(image)
```

![Result](readme_image.png)

### Size Control

The output width and height paramers can be used to specify the *maximum*
width and height of the rendered image. The image will be rendered with
the largest possible size that fits within the given bounding box and keeps
the original image's aspect ratio.

To ensure an image with exact dimension, use the 'fill' keyword option to
add add a background fill extending the image to the specified size.

Note that the width and height is measured in termianl characters, which
are not square but rectangular.

```python
from ansi_image import AnsiImage

image = AnsiImage.from_file("tests/test_image.png")

# Use the current terminal size as default width and height
print(image)
input("press enter...")

# Set an explicit max width and height
render = image.render(max_width=80, max_height=24)
print(render)
input("press enter...")

# Using format strings
print(f"{image:w=40,h=20}")
print(f"{image:width=60}")
input("press enter...")

# Add background color to fill the entire bounding box
rendered = image.render(fill="#ffffff")
print(rendered)
print(f"rendered image with dimensions f{rendered.width}x{rendered.height}")
```

### Image Stretching

For precise control over image dimensions without maintaining aspect ratio, you can manipulate the image using PIL before rendering:

```python
from ansi_image import AnsiImage
from PIL import Image

# Load and stretch the image to exact dimensions
img = Image.open("tests/test_image.png")
stretched_img = img.resize((160, 48))  # Stretch to exact dimensions
ansi_stretched = AnsiImage.from_image(stretched_img)
print(ansi_stretched)
```

### Command Line Tool

The package also includes a command-line tool:

```bash
uvx ansi-image tests/test_image.png
uvx ansi-image --width 80 --height 24 tests/test_image.png
# From the source repository
uv run ansi-image tests/test_image.png
```

## API Reference

### AnsiImage

Main class that stores the original PIL Image and provides rendering methods.

- `img = render(max_width=None, max_height=None, flags=0, fill=None)` - Render to RenderedAnsiImage
- `AnsiImage.from_image(img, ...)` - Static method to create directly from PIL Image
- `AnsiImage.from_file(path, ...)` - Static method to load and create from file
- `str(img)` - Convert to printable string

### RenderedAnsiImage

Contains the pre-rendered text representation that can be printed.

- `str(rendered)` - Convert to printable string
- `rendered.width` - Width in terminal columns
- `rendered.height` - Height in terminal rows
- `rendered.data` - List of strings with ANSI codes, one per row.

## Memory Usage

The `AnsiImage` object keeps the full PIL Image in memory, allowing multiple
renders with different parameters.

The `RenderedAnsiImage` objects only contain the text representation. Use the
`render()` method to obtain the latter and discard the former if memory usage
is a concern.

## Algorithm

The rendering algorithm is a direct port from the C++ implementation in [TerminalImageViewer](https://github.com/stefanhaustein/TerminalImageViewer), providing the same high-quality terminal image display in pure Python.

On a high-level, it works by splitting the image into 4x8 pixel blocks and
selecting the most appropriate unicode block character for each, with the
closest matching foreground and background colors.

## Installation

Install via pypi:

```bash
pip install ansi-image
```

Or use as a standalone tool:

```bash
uvx ansi-image tests/test_image.png
```