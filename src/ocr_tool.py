import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.image_inspector import (
    DEFAULT_MODEL,
    encode_image_as_data_url,
)
from src.schemas import (
    OCRExtraction,
    OCRTextSpan,
)


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparison.

    The normalization:
    - makes Unicode characters consistent;
    - ignores uppercase/lowercase differences;
    - ignores punctuation;
    - collapses repeated whitespace.

    Examples:
        "28th St." -> "28th st"
        "28TH ST"  -> "28th st"
        " 28th  St " -> "28th st"
    """

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = normalized.lower().strip()

    normalized = normalized.replace(
        "_",
        " ",
    )

    normalized = re.sub(
        r"[^\w\s]",
        " ",
        normalized,
        flags=re.UNICODE,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def reconcile_text_targets(
    text_targets: List[str],
    detected_text: List[OCRTextSpan],
) -> Tuple[List[str], List[str]]:
    """
    Compare requested targets with OCR text deterministically.

    Capitalization, punctuation, and extra whitespace are ignored.
    """

    target_matches: List[str] = []
    target_mismatches: List[str] = []

    detected_pairs = []

    for text_span in detected_text:
        detected_pairs.append(
            (
                text_span.text,
                normalize_text(text_span.text),
            )
        )

    for target in text_targets:
        normalized_target = normalize_text(
            target
        )

        matching_text: Optional[str] = None

        for original_text, normalized_detected in detected_pairs:
            if (
                normalized_target
                and normalized_target
                == normalized_detected
            ):
                matching_text = original_text
                break

        if matching_text is not None:
            target_matches.append(
                f'Target "{target}" matches detected text '
                f'"{matching_text}" after normalization.'
            )
        else:
            target_mismatches.append(
                f'Target "{target}" was not found in '
                "the detected image text."
            )

    return (
        target_matches,
        target_mismatches,
    )


def extract_text_from_image(
    image_path: str,
    claim: str,
    text_targets: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
) -> OCRExtraction:
    """
    Extract visible text from an image.

    The model performs transcription. Target matching is then
    recalculated deterministically in Python.
    """

    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Check the .env file in the project root."
        )

    targets = (
        text_targets
        if text_targets is not None
        else []
    )

    image_data_url = encode_image_as_data_url(
        image_path
    )

    client = OpenAI(
        api_key=api_key
    )

    targets_json = json.dumps(
        targets,
        ensure_ascii=False,
    )

    system_prompt = """
You are the OCR Tool inside a multimodal evidence
verification agent.

Your job is to transcribe readable text that is visibly
present in the supplied image.

Rules:

1. Transcribe visible text as exactly as possible.
2. Do not guess missing or unreadable characters.
3. Do not use outside knowledge.
4. Do not make the final supported, refuted, or insufficient
   decision.
5. Record the approximate location of every detected text span.
6. Lower confidence when text is small, blurry, distorted,
   hidden, or ambiguous.
7. Give high relevance_to_claim only when a text span is
   directly useful for checking the supplied claim.
8. Sentence punctuation in the claim is not necessarily part
   of the visible text.
9. Ignore text inferred only from object identity.
10. Report uncertainty instead of inventing text.

Focus primarily on transcription. Target matching will be
recalculated later by deterministic program logic.
""".strip()

    user_prompt = f"""
Claim:
{claim}

Requested text targets:
{targets_json}

Inspect the image carefully and return structured OCR evidence.

Include text relevant to the claim and other clearly readable
text. Do not make the final verification decision.
""".strip()

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high",
                    },
                ],
            },
        ],
        text_format=OCRExtraction,
    )

    model_extraction = response.output_parsed

    if model_extraction is None:
        raise RuntimeError(
            "The OCR model returned no parsed result."
        )

    if targets:
        (
            deterministic_matches,
            deterministic_mismatches,
        ) = reconcile_text_targets(
            text_targets=targets,
            detected_text=model_extraction.detected_text,
        )
    else:
        deterministic_matches = []
        deterministic_mismatches = []

    final_extraction = OCRExtraction(
        detected_text=model_extraction.detected_text,
        target_matches=deterministic_matches,
        target_mismatches=deterministic_mismatches,
        uncertainty_notes=model_extraction.uncertainty_notes,
    )

    return final_extraction