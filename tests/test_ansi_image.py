import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from ansi_image.ansi_image import AnsiImage, RenderedAnsiImage

class TestAnsiImage(unittest.TestCase):
    """Test suite for AnsiImage conversion functionality."""
    
    test_image_path: Path
    expected_outputs_path: Path
    expected_outputs: Dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        """Load expected outputs from JSON file."""
        cls.test_image_path = Path(__file__).parent / "test_image.png"
        cls.expected_outputs_path = Path(__file__).parent / "expected_outputs.json"
        
        if not cls.test_image_path.exists():
            raise FileNotFoundError(f"Test image not found: {cls.test_image_path}")
        
        if not cls.expected_outputs_path.exists():
            raise FileNotFoundError(f"Expected outputs not found: {cls.expected_outputs_path}")
        
        with open(cls.expected_outputs_path, 'r') as f:
            cls.expected_outputs = json.load(f)

    def test_default_40x20_output(self) -> None:
        """Test conversion with default settings at 40x20."""
        ansi_img = AnsiImage.from_file(str(self.test_image_path), 40, 20, 0)
        expected = self.expected_outputs['default_40x20']
        
        self.assertEqual(ansi_img.width, expected['width'])
        self.assertEqual(ansi_img.height, expected['height'])
        self.assertEqual(ansi_img.data, expected['data'])

    def test_small_20x10_output(self) -> None:
        """Test conversion with small dimensions 20x10."""
        ansi_img = AnsiImage.from_file(str(self.test_image_path), 20, 10, 0)
        expected = self.expected_outputs['small_20x10']
        
        self.assertEqual(ansi_img.width, expected['width'])
        self.assertEqual(ansi_img.height, expected['height'])
        self.assertEqual(ansi_img.data, expected['data'])

    def test_large_60x30_output(self) -> None:
        """Test conversion with large dimensions 60x30."""
        ansi_img = AnsiImage.from_file(str(self.test_image_path), 60, 30, 0)
        expected = self.expected_outputs['large_60x30']
        
        self.assertEqual(ansi_img.width, expected['width'])
        self.assertEqual(ansi_img.height, expected['height'])
        self.assertEqual(ansi_img.data, expected['data'])

    def test_auto_size_output(self) -> None:
        """Test conversion with automatic terminal size detection."""
        # Mock os.get_terminal_size to return consistent dimensions
        mock_terminal_size = os.terminal_size((72, 24))
        with patch('ansi_image.ansi_image.os.get_terminal_size', return_value=mock_terminal_size):
            ansi_img = AnsiImage.from_file(str(self.test_image_path), None, None, 0)
            expected = self.expected_outputs['auto_size']
            
            self.assertEqual(ansi_img.width, expected['width'])
            self.assertEqual(ansi_img.height, expected['height'])
            self.assertEqual(ansi_img.data, expected['data'])

    def test_noopt_flag_output(self) -> None:
        """Test conversion with FLAG_NOOPT (simple mode)."""
        ansi_img = AnsiImage.from_file(str(self.test_image_path), 40, 20, 1)  # FLAG_NOOPT = 1
        expected = self.expected_outputs['noopt_40x20']
        
        self.assertEqual(ansi_img.width, expected['width'])
        self.assertEqual(ansi_img.height, expected['height'])
        self.assertEqual(ansi_img.data, expected['data'])

    def test_scaling_dimensions_only(self) -> None:
        """Test that scaling works correctly by checking only dimensions."""
        test_cases = [
            (10, 5),
            (80, 40),
            (100, 50),
            (5, 3),
        ]
        
        for width, height in test_cases:
            with self.subTest(width=width, height=height):
                ansi_img = AnsiImage.from_file(str(self.test_image_path), width, height, 0)
                
                self.assertLessEqual(ansi_img.width, width, 
                    f"Width {ansi_img.width} should be <= requested {width}")
                self.assertLessEqual(ansi_img.height, height,
                    f"Height {ansi_img.height} should be <= requested {height}")
                
                self.assertGreater(ansi_img.width, 0, "Width should be > 0")
                self.assertGreater(ansi_img.height, 0, "Height should be > 0")
                self.assertEqual(len(ansi_img.data), ansi_img.height, 
                    "Number of data lines should match height")

    def test_aspect_ratio_preservation(self) -> None:
        """Test that aspect ratio is reasonably preserved during scaling."""
        from PIL import Image
        with Image.open(self.test_image_path) as img:
            original_width, original_height = img.size
            original_aspect = original_width / original_height

        # Note: smaller targets may have more aspect ratio deviation due to discrete character cells
        test_cases = [
            (40, 20, 0.1),   # Should be very close
            (80, 40, 0.1),   # Should be very close
            (20, 10, 0.2),   # Allow more tolerance for small sizes
        ]
        
        for target_width, target_height, tolerance in test_cases:
            with self.subTest(target_width=target_width, target_height=target_height):
                ansi_img = AnsiImage.from_file(str(self.test_image_path), target_width, target_height, 0)
                
                actual_pixel_width = ansi_img.width * 4
                actual_pixel_height = ansi_img.height * 8
                actual_aspect = actual_pixel_width / actual_pixel_height
                
                self.assertAlmostEqual(actual_aspect, original_aspect, delta=tolerance,
                    msg=f"Aspect ratio should be reasonably preserved. Original: {original_aspect:.3f}, "
                        f"Actual: {actual_aspect:.3f}, Target: {target_width}x{target_height}")

    def test_dimension_validation(self) -> None:
        """Test single dimension specification with aspect ratio preservation."""
        ansi_img_width_only = AnsiImage.from_file(str(self.test_image_path), 40, None, 0)
        self.assertGreater(ansi_img_width_only.width, 0)
        self.assertGreater(ansi_img_width_only.height, 0)
        self.assertLessEqual(ansi_img_width_only.width, 40)
        
        ansi_img_height_only = AnsiImage.from_file(str(self.test_image_path), None, 20, 0)
        self.assertGreater(ansi_img_height_only.width, 0)
        self.assertGreater(ansi_img_height_only.height, 0)
        self.assertLessEqual(ansi_img_height_only.height, 20)

    def test_string_representation(self) -> None:
        """Test string representation methods."""
        ansi_img = AnsiImage.from_file(str(self.test_image_path), 20, 10, 0)
        
        str_repr = str(ansi_img)
        self.assertEqual(str_repr, "\n".join(ansi_img.data))
        
        repr_str = repr(ansi_img)
        self.assertIn("RenderedAnsiImage", repr_str)
        self.assertIn(f"width={ansi_img.width}", repr_str)
        self.assertIn(f"height={ansi_img.height}", repr_str)

    def test_from_pil_image(self) -> None:
        """Test creating RenderedAnsiImage from PIL Image object."""
        from PIL import Image
        
        with Image.open(self.test_image_path) as img:
            ansi_img = AnsiImage.from_image(img, 30, 15, 0)
            
            self.assertGreater(ansi_img.width, 0)
            self.assertGreater(ansi_img.height, 0)
            self.assertEqual(len(ansi_img.data), ansi_img.height)

    def test_new_ansi_image_with_render(self) -> None:
        """Test the new AnsiImage class with render method."""
        from PIL import Image
        
        with Image.open(self.test_image_path) as img:
            ansi_img = AnsiImage(img)
            
            rendered = ansi_img.render(30, 15, 0)
            
            self.assertIsInstance(rendered, RenderedAnsiImage)
            self.assertGreater(rendered.width, 0)
            self.assertGreater(rendered.height, 0)
            self.assertEqual(len(rendered.data), rendered.height)
            
            rendered_large = ansi_img.render(60, 30, 0)
            self.assertGreater(rendered_large.width, rendered.width)
            self.assertGreater(rendered_large.height, rendered.height)

    def test_different_flag_combinations(self) -> None:
        """Test different flag combinations for rendering options."""
        flag_values = [0, 1, 2, 4]
        
        for flags in flag_values:
            with self.subTest(flags=flags):
                try:
                    ansi_img = AnsiImage.from_file(str(self.test_image_path), 20, 10, flags)
                    
                    self.assertGreater(ansi_img.width, 0)
                    self.assertGreater(ansi_img.height, 0)
                    self.assertEqual(len(ansi_img.data), ansi_img.height)
                    
                    for line in ansi_img.data:
                        self.assertIsInstance(line, str)
                        self.assertIn('\x1b', line)
                
                except Exception as e:
                    self.assertTrue(isinstance(e, (ValueError, KeyError)), 
                        f"Unexpected exception type for flags {flags}: {type(e)}")


if __name__ == '__main__':
    unittest.main()