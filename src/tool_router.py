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
        The sign says "28th St."
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


def keyword_is_present(
    claim: str,
    keyword: str,
) -> bool:
    """
    Match one complete word or phrase.

    This prevents a keyword such as "sign" from accidentally
    matching a longer word such as "signal".
    """

    escaped_keyword = re.escape(
        keyword
    )

    escaped_keyword = escaped_keyword.replace(
        r"\ ",
        r"\s+",
    )

    pattern = (
        rf"(?<!\w)"
        rf"{escaped_keyword}"
        rf"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            claim,
            flags=re.IGNORECASE,
        )
    )


def find_matching_keywords(
    claim: str,
) -> List[str]:
    """Find text-related keywords in a claim."""

    return [
        keyword
        for keyword in TEXT_KEYWORDS
        if keyword_is_present(
            claim,
            keyword,
        )
    ]


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

    The Image Inspector is always used.

    OCR is added when the claim depends on visible writing,
    numbers, labels, signs, dates, or readable information.
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