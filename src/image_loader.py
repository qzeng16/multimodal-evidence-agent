from pathlib import Path
from typing import Dict, Union

from PIL import Image, UnidentifiedImageError


ImageMetadata = Dict[str, Union[str, int]]


def load_image_metadata(image_path: str) -> ImageMetadata:
    """
    Validate an image file and return its basic metadata.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary containing image metadata.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file cannot be opened as an image.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Image path is not a file: {image_path}"
        )

    try:
        with Image.open(path) as image:
            image.load()

            return {
                "image_path": str(path),
                "file_name": path.name,
                "format": image.format or "unknown",
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }

    except UnidentifiedImageError as error:
        raise ValueError(
            f"Unsupported or invalid image file: {image_path}"
        ) from error

    except OSError as error:
        raise ValueError(
            f"Failed to open image file: {image_path}"
        ) from error