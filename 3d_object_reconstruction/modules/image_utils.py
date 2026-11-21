from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def inspect_image_bytes(data: bytes, filename: str) -> dict:
    """Inspect an uploaded image without writing it to disk."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Only JPG, JPEG, and PNG files are supported.")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            image = image.convert("RGB")
            array = np.asarray(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("File is not a readable image.") from exc

    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings: list[str] = []
    if min(width, height) < 600:
        warnings.append("Low resolution for detailed reconstruction")
    if blur_score < 50:
        warnings.append("Possibly blurry (Laplacian variance below 50)")
    return {
        "filename": Path(filename).name,
        "valid": True,
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 2),
        "warnings": "; ".join(warnings) if warnings else "None",
    }


def save_uploaded_images(uploaded_files: Iterable, destination: Path) -> tuple[list[Path], list[str]]:
    """Validate and safely save Streamlit UploadedFile-like objects."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rejected: list[str] = []
    existing_hashes = set()
    for existing in destination.iterdir():
        if existing.is_file() and existing.suffix.lower() in ALLOWED_SUFFIXES:
            try:
                existing_hashes.add(hashlib.sha256(existing.read_bytes()).hexdigest())
            except OSError:
                pass

    for index, uploaded in enumerate(uploaded_files, start=1):
        original_name = Path(uploaded.name).name
        try:
            data = uploaded.getvalue()
            inspect_image_bytes(data, original_name)
            file_hash = hashlib.sha256(data).hexdigest()
            if file_hash in existing_hashes:
                rejected.append(f"{original_name}: duplicate image skipped")
                continue
            existing_hashes.add(file_hash)
            suffix = Path(original_name).suffix.lower()
            stem = Path(original_name).stem
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_-") or f"image_{index}"
            target = destination / f"{index:04d}_{safe_stem}{suffix}"
            target.write_bytes(data)
            saved.append(target)
        except Exception as exc:
            rejected.append(f"{original_name}: {exc}")

    return saved, rejected


def preprocess_image(input_path: Path, output_path: Path | None = None, max_dimension: int = 2400) -> np.ndarray:
    """Optional grayscale preprocessing helper for experiments, not the default COLMAP inputs."""
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest > max_dimension:
        scale = max_dimension / largest
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise IOError(f"Could not write preprocessed image: {output_path}")
    return image
