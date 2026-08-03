import re
from typing import List

from src.schemas import ToolRoutingDecision


TEXT_KEYWORDS = [
    "text",
    "word",
    "words",
    "letter",
    "letters",
    "number",
    "numbers",
    "sign",
    "signage",
    "label",
    "logo",
    "license plate",
    "plate",
    "street name",
    "written",
    "writes",
    "says",
    "reads",
    "printed",
    "spelled",
    "date",
    "year",
    "price",
    "address",
]


def find_text_targets(
    claim: str,
) -> List[str]:
    """
    Extract text enclosed in single or double quotation marks.

    Example:
        The street sign says "28th St."
        -> ["28th St."]
    """

    double_quoted_targets = re.findall(
        r'"([^"]+)"',
        claim,
    )

    single_quoted_targets = re.findall(
        r"'([^']+)'",
        claim,
    )

    return (
        double_quoted_targets
        + single_quoted_targets
    )


def find_matching_keywords(
    claim: str,
) -> List[str]:
    """
    Find text-related keywords in a claim.

    The comparison is case-insensitive.
    """

    normalized_claim = claim.lower()

    matched_keywords = [
        keyword
        for keyword in TEXT_KEYWORDS
        if keyword in normalized_claim
    ]

    return matched_keywords


def contains_year_like_number(
    claim: str,
) -> bool:
    """
    Detect a four-digit year-like number.

    Examples:
        2024 -> True
        1998 -> True
        28 -> False
    """

    year_matches = re.findall(
        r"\b(?:19|20)\d{2}\b",
        claim,
    )

    return bool(year_matches)


def route_tools(
    claim: str,
) -> ToolRoutingDecision:
    """
    Select tools for one verification claim.

    The Image Inspector is always used because the system
    verifies claims against an image.

    OCR is added when the claim depends on visible writing,
    numbers, labels, signs, dates, or other readable content.
    """

    matched_keywords = find_matching_keywords(
        claim
    )

    text_targets = find_text_targets(
        claim
    )

    has_year = contains_year_like_number(
        claim
    )

    use_ocr = bool(
        matched_keywords
        or text_targets
        or has_year
    )

    if use_ocr:
        reasoning = (
            "The claim depends on visible text, numbers, "
            "signage, labels, dates, or written information. "
            "The system should use both the Image Inspector "
            "and the OCR tool."
        )
    else:
        reasoning = (
            "The claim concerns visual properties rather "
            "than readable text. The Image Inspector is "
            "sufficient, so OCR is not required."
        )

    return ToolRoutingDecision(
        use_image_inspector=True,
        use_ocr=use_ocr,
        reasoning=reasoning,
        matched_keywords=matched_keywords,
        text_targets=text_targets,
    ) 