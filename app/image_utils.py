from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


register_heif_opener()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".heic", ".heif")
JPEG_FORMATS = {"JPEG", "JPG"}


def is_jpeg_image(image_path: str | Path) -> bool:
    with Image.open(image_path) as img:
        return (img.format or "").upper() in JPEG_FORMATS


def save_image_as_jpeg(
    source_path: str | Path,
    target_path: str | Path,
    *,
    max_resolution: int = 0,
    quality: int = 90,
) -> None:
    with Image.open(source_path) as img:
        img = ImageOps.exif_transpose(img)

        if max_resolution > 0 and max(img.size) > max_resolution:
            img.thumbnail(
                (max_resolution, max_resolution),
                getattr(Image, "Resampling", Image).LANCZOS,
            )

        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(target_path, format="JPEG", quality=quality, optimize=True)
