import unittest
from io import BytesIO

from PIL import Image

from app.vision_image import prepare_image


class VisionImageTests(unittest.TestCase):
    def test_prepare_image_downsizes_large_image(self) -> None:
        img = Image.new("RGB", (4000, 2000), color=(255, 0, 0))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        data = buffer.getvalue()

        processed, (width, height), mime = prepare_image(data, max_edge=1280)

        self.assertLessEqual(max(width, height), 1280)
        self.assertEqual(mime, "image/jpeg")
        self.assertTrue(len(processed) < len(data))


if __name__ == "__main__":
    unittest.main()
