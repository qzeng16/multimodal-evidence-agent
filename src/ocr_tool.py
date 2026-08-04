import base64
import io
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from src.image_inspector import DEFAULT_MODEL
from src.schemas import (
    OCRExtraction,
    OCRTextSpan,
)


OCR_REGION_CONFIG_PATH = Path(
    "data/ocr_regions.json"
)

OCR_DEBUG_DIRECTORY = Path(
    "outputs/ocr_views"
)


class OCRRegionConfig(BaseModel):
    """Normalized region used for OCR preprocessing."""

    x_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    x_max: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_max: float = Field(
        ge=0.0,
        le=1.0,
    )

    description: str

    rotations: List[int] = Field(
        default_factory=lambda: [
            0,
            90,
            270,
        ]
    )

    upscale_factor: int = Field(
        default=3,
        ge=1,
        le=4,
    )


class OCRView(BaseModel):
    """Metadata for one generated OCR view."""

    name: str
    description: str


class BlindOCRTextSpan(BaseModel):
    """
    One consolidated text span produced without access
    to the verification claim or expected answer.
    """

    text: str
    location: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    source_views: List[str] = Field(
        default_factory=list
    )


class BlindOCRResult(BaseModel):
    """Claim-independent OCR output."""

    detected_text: List[
        BlindOCRTextSpan
    ] = Field(
        default_factory=list
    )

    uncertainty_notes: List[str] = Field(
        default_factory=list
    )


def normalize_text(
    text: str,
) -> str:
    """
    Normalize text for deterministic comparison.

    This ignores:

    - capitalization
    - ordinary punctuation
    - apostrophe differences
    - repeated whitespace
    - common Unicode variations
    """

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = normalized.lower()

    normalized = normalized.replace(
        "’",
        "",
    )

    normalized = normalized.replace(
        "'",
        "",
    )

    normalized = normalized.replace(
        "_",
        " ",
    )

    normalized = re.sub(
        r"[^\w\s]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def compact_text(
    text: str,
) -> str:
    """
    Create a comparison form without whitespace.

    This allows minor segmentation differences such as:

    MAM'S
    MAM S
    MAMS
    """

    return normalize_text(
        text
    ).replace(
        " ",
        "",
    )


def deduplicate_targets(
    targets: Sequence[str],
) -> List[str]:
    """Remove duplicate targets while preserving order."""

    unique_targets: List[str] = []
    seen = set()

    for target in targets:
        cleaned_target = target.strip()

        normalized_target = compact_text(
            cleaned_target
        )

        if not normalized_target:
            continue

        if normalized_target in seen:
            continue

        seen.add(
            normalized_target
        )

        unique_targets.append(
            cleaned_target
        )

    return unique_targets


def load_region_config(
    image_path: str,
) -> Optional[OCRRegionConfig]:
    """Load an optional OCR region for one image."""

    if not OCR_REGION_CONFIG_PATH.exists():
        return None

    try:
        raw_config = json.loads(
            OCR_REGION_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid JSON in "
            f"{OCR_REGION_CONFIG_PATH}: "
            f"{error}"
        ) from error

    normalized_image_path = Path(
        image_path
    ).as_posix()

    image_config = raw_config.get(
        normalized_image_path
    )

    if image_config is None:
        return None

    region_config = (
        OCRRegionConfig.model_validate(
            image_config
        )
    )

    if (
        region_config.x_min
        >= region_config.x_max
    ):
        raise ValueError(
            "OCR region x_min must be "
            "smaller than x_max."
        )

    if (
        region_config.y_min
        >= region_config.y_max
    ):
        raise ValueError(
            "OCR region y_min must be "
            "smaller than y_max."
        )

    valid_rotations = []

    for rotation in region_config.rotations:
        normalized_rotation = (
            rotation % 360
        )

        if normalized_rotation not in {
            0,
            90,
            180,
            270,
        }:
            raise ValueError(
                "OCR rotations must be "
                "0, 90, 180, or 270 degrees."
            )

        if normalized_rotation not in (
            valid_rotations
        ):
            valid_rotations.append(
                normalized_rotation
            )

    region_config.rotations = (
        valid_rotations
    )

    return region_config


def calculate_crop_box(
    image: Image.Image,
    region: OCRRegionConfig,
) -> Tuple[int, int, int, int]:
    """Convert a normalized region into pixel coordinates."""

    width, height = image.size

    left = round(
        region.x_min * width
    )

    top = round(
        region.y_min * height
    )

    right = round(
        region.x_max * width
    )

    bottom = round(
        region.y_max * height
    )

    left = max(
        0,
        min(left, width - 1),
    )

    top = max(
        0,
        min(top, height - 1),
    )

    right = max(
        left + 1,
        min(right, width),
    )

    bottom = max(
        top + 1,
        min(bottom, height),
    )

    return (
        left,
        top,
        right,
        bottom,
    )


def upscale_image(
    image: Image.Image,
    factor: int,
) -> Image.Image:
    """Upscale one OCR image using Lanczos resampling."""

    if factor <= 1:
        return image.copy()

    width, height = image.size

    return image.resize(
        (
            width * factor,
            height * factor,
        ),
        resample=(
            Image.Resampling.LANCZOS
        ),
    )


def create_high_contrast_view(
    image: Image.Image,
) -> Image.Image:
    """Create a grayscale high-contrast OCR view."""

    grayscale_image = (
        ImageOps.grayscale(
            image
        )
    )

    contrast_image = (
        ImageOps.autocontrast(
            grayscale_image
        )
    )

    return contrast_image.convert(
        "RGB"
    )


def save_debug_views(
    image_path: str,
    views: Sequence[
        Tuple[OCRView, Image.Image]
    ],
) -> None:
    """Save generated OCR views for inspection."""

    image_stem = Path(
        image_path
    ).stem

    output_directory = (
        OCR_DEBUG_DIRECTORY
        / image_stem
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for view, image in views:
        output_path = (
            output_directory
            / f"{view.name}.png"
        )

        image.save(
            output_path,
            format="PNG",
        )


def build_ocr_views(
    image_path: str,
) -> List[
    Tuple[OCRView, Image.Image]
]:
    """
    Generate full-image and region-specific OCR views.

    When no region annotation exists, only the original
    full image is used.
    """

    path = Path(
        image_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Image path is not a file: "
            f"{image_path}"
        )

    try:
        with Image.open(path) as raw_image:
            source_image = (
                ImageOps.exif_transpose(
                    raw_image
                )
                .convert("RGB")
            )

    except Exception as error:
        raise ValueError(
            f"Unable to open image: "
            f"{image_path}"
        ) from error

    views: List[
        Tuple[OCRView, Image.Image]
    ] = [
        (
            OCRView(
                name="full_original",
                description=(
                    "The complete original image."
                ),
            ),
            source_image.copy(),
        )
    ]

    region_config = load_region_config(
        image_path
    )

    if region_config is None:
        save_debug_views(
            image_path=image_path,
            views=views,
        )

        return views

    crop_box = calculate_crop_box(
        image=source_image,
        region=region_config,
    )

    region_crop = source_image.crop(
        crop_box
    )

    region_crop = upscale_image(
        image=region_crop,
        factor=(
            region_config.upscale_factor
        ),
    )

    views.append(
        (
            OCRView(
                name="region_upscaled",
                description=(
                    f"An enlarged crop of: "
                    f"{region_config.description}"
                ),
            ),
            region_crop.copy(),
        )
    )

    for rotation in (
        region_config.rotations
    ):
        if rotation == 0:
            continue

        rotated_image = (
            region_crop.rotate(
                rotation,
                expand=True,
            )
        )

        views.append(
            (
                OCRView(
                    name=(
                        f"region_rot"
                        f"{rotation}"
                    ),
                    description=(
                        "The same enlarged crop "
                        f"rotated {rotation} degrees."
                    ),
                ),
                rotated_image,
            )
        )

    contrast_image = (
        create_high_contrast_view(
            region_crop
        )
    )

    preferred_rotation = (
        270
        if 270
        in region_config.rotations
        else 0
    )

    if preferred_rotation != 0:
        contrast_image = (
            contrast_image.rotate(
                preferred_rotation,
                expand=True,
            )
        )

    views.append(
        (
            OCRView(
                name=(
                    "region_high_contrast"
                    f"_rot{preferred_rotation}"
                ),
                description=(
                    "A high-contrast grayscale "
                    "version of the enlarged text "
                    "region."
                ),
            ),
            contrast_image,
        )
    )

    save_debug_views(
        image_path=image_path,
        views=views,
    )

    return views


def image_to_data_url(
    image: Image.Image,
) -> str:
    """Convert an in-memory image into a PNG data URL."""

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return (
        "data:image/png;base64,"
        f"{encoded_image}"
    )


def spans_match_target(
    detected_text: str,
    target: str,
) -> bool:
    """Compare one OCR span with one target."""

    normalized_detected = (
        normalize_text(
            detected_text
        )
    )

    normalized_target = (
        normalize_text(
            target
        )
    )

    if (
        normalized_detected
        == normalized_target
    ):
        return True

    return (
        compact_text(
            detected_text
        )
        == compact_text(
            target
        )
    )


def convert_blind_spans(
    blind_spans: Sequence[
        BlindOCRTextSpan
    ],
    text_targets: Sequence[str],
) -> List[OCRTextSpan]:
    """
    Convert blind OCR output into the project schema.

    Target relevance is added locally after OCR completes.
    """

    converted_spans: List[
        OCRTextSpan
    ] = []

    for span in blind_spans:
        matches_target = any(
            spans_match_target(
                detected_text=span.text,
                target=target,
            )
            for target in text_targets
        )

        if matches_target:
            relevance = 1.0
        elif text_targets:
            relevance = 0.5
        else:
            relevance = 0.0

        source_view_text = (
            ", ".join(
                span.source_views
            )
            if span.source_views
            else "unspecified view"
        )

        location = (
            f"{span.location}; "
            f"source views: "
            f"{source_view_text}"
        )

        converted_spans.append(
            OCRTextSpan(
                text=span.text,
                location=location,
                confidence=span.confidence,
                relevance_to_claim=relevance,
            )
        )

    return converted_spans


def reconcile_text_targets(
    text_targets: Sequence[str],
    detected_text: Sequence[
        OCRTextSpan
    ],
) -> Tuple[List[str], List[str]]:
    """
    Deterministically compare targets with blind OCR output.
    """

    unique_targets = deduplicate_targets(
        text_targets
    )

    target_matches: List[str] = []
    target_mismatches: List[str] = []

    for target in unique_targets:
        matching_span = None

        for span in detected_text:
            if spans_match_target(
                detected_text=span.text,
                target=target,
            ):
                matching_span = span
                break

        if matching_span is not None:
            target_matches.append(
                f'Target "{target}" matches '
                f'detected text '
                f'"{matching_span.text}" '
                f"after normalization."
            )

            continue

        if detected_text:
            detected_preview = ", ".join(
                f'"{span.text}"'
                for span in detected_text[:5]
            )

            target_mismatches.append(
                f'Target "{target}" was not '
                f"found in the multi-view blind "
                f"OCR transcription. Detected "
                f"text included: "
                f"{detected_preview}."
            )

        else:
            target_mismatches.append(
                f'Target "{target}" was not '
                f"found because multi-view blind "
                f"OCR detected no readable text."
            )

    return (
        target_matches,
        target_mismatches,
    )


def extract_text_from_image(
    image_path: str,
    claim: Optional[str] = None,
    text_targets: Optional[
        Sequence[str]
    ] = None,
    model: str = DEFAULT_MODEL,
) -> OCRExtraction:
    """
    Run multi-view, claim-independent OCR.

    The claim and text targets are never sent to the model.
    Text targets are compared locally only after the blind
    transcription has been returned.
    """

    # Kept only for compatibility with the current main.py.
    # It is deliberately excluded from all API inputs.
    _ = claim

    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Check the .env file in the "
            "project root."
        )

    views = build_ocr_views(
        image_path
    )

    client = OpenAI(
        api_key=api_key
    )

    system_prompt = """
You are a blind multi-view OCR transcription tool.

You receive several views derived from the same source image.
Some views may be:

- the complete original image
- a cropped text region
- an enlarged version
- a rotated version
- a high-contrast version

You do not know the user's claim, target phrase, expected answer,
gold label, or verification task.

Your only job is to independently transcribe text that is
actually visible.

Rules:

1. Never guess wording from context, familiarity, restaurant
   names, brands, or likely phrases.
2. Never complete unreadable letters into a familiar name.
3. Use the clearest available view for each physical text region.
4. Consolidate repeated views of the same physical text into one
   final text span.
5. Do not return different guesses as separate spans when they
   refer to the same physical writing.
6. When views disagree, choose only what is visually supported
   and lower confidence.
7. Preserve visible spelling as closely as possible.
8. Do not add punctuation that is not visibly supported.
9. If only part of a phrase is readable, return only that part.
10. Treat curved, rotated, blurred, distant, low-contrast, or
    partially hidden text cautiously.
11. List which source views support each transcription.
12. Record important ambiguity in uncertainty_notes.
13. Do not verify or discuss any claim.
""".strip()

    content = [
        {
            "type": "input_text",
            "text": (
                "The following images are different "
                "views of the same source image. "
                "Independently consolidate all readable "
                "text across the views."
            ),
        }
    ]

    for view, image in views:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"View name: {view.name}\n"
                    f"View description: "
                    f"{view.description}"
                ),
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": (
                    image_to_data_url(
                        image
                    )
                ),
                "detail": "high",
            }
        )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        text_format=BlindOCRResult,
    )

    blind_result = (
        response.output_parsed
    )

    if blind_result is None:
        raise RuntimeError(
            "The OCR model returned no "
            "parsed result."
        )

    targets = list(
        text_targets or []
    )

    converted_text = convert_blind_spans(
        blind_spans=(
            blind_result.detected_text
        ),
        text_targets=targets,
    )

    (
        target_matches,
        target_mismatches,
    ) = reconcile_text_targets(
        text_targets=targets,
        detected_text=converted_text,
    )

    view_names = [
        view.name
        for view, _ in views
    ]

    preprocessing_note = (
        "Multi-view blind OCR used: "
        + ", ".join(view_names)
        + "."
    )

    return OCRExtraction(
        detected_text=converted_text,
        target_matches=target_matches,
        target_mismatches=target_mismatches,
        uncertainty_notes=[
            preprocessing_note,
            *blind_result.uncertainty_notes,
        ],
    )