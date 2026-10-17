"""JPEG / HEIC helpers shared by the Telegram layer and the tools.

``IMAGE_EXTENSIONS`` is the set of picture formats the bot accepts as uploads (photos
arrive as JPEG; a document with one of these extensions is routed to the image job and
converted). Everything is normalized to a JPEG copy with ``save_image_as_jpeg`` so the
rest of the code deals with one format; transparent pictures are flattened on white
rather than on the black that a bare ``convert("RGB")`` would give.
"""
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


register_heif_opener()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp", ".gif", ".bmp")
JPEG_FORMATS = {"JPEG", "JPG"}


def is_jpeg_image(image_path: str | Path) -> bool:
    with Image.open(image_path) as img:
        return (img.format or "").upper() in JPEG_FORMATS


def flatten_to_rgb(img: Image.Image) -> Image.Image:
    """An RGB copy; transparent areas (RGBA, LA, palette with transparency) become white."""
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img if img.mode == "RGB" else img.convert("RGB")


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

        img = flatten_to_rgb(img)

        img.save(target_path, format="JPEG", quality=quality, optimize=True)
