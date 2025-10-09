from __future__ import annotations

from io import BytesIO
from typing import Tuple

from PIL import Image


def prepare_image(data: bytes, max_edge: int) -> tuple[bytes, Tuple[int, int], str]:
    """Return resized JPEG bytes, resulting dimensions, and mime type."""
    with Image.open(BytesIO(data)) as img:
        img = img.convert("RGB")
        width, height = img.size
        max_dimension = max(width, height)
        if max_edge > 0 and max_dimension > max_edge:
            scale = max_edge / float(max_dimension)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            img = img.resize(new_size, Image.LANCZOS)
        output = BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue(), img.size, "image/jpeg"
