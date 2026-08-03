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


def find_quoted_text_targets(
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


def find_year_targets(
    claim: str,
) -> List[str]:
    """
    Extract four-digit year-like values from a claim.

    Examples:
        "installed in 2024" -> ["2024"]
        "built in 1998" -> ["1998"]
    """

    return re.findall(
        r"\b(?:19|20)\d{2}\b",
        claim,
    )


def deduplicate_targets(
    targets: List[str],
) -> List[str]:
    """Remove duplicate targets while preserving order."""

    unique_targets: List[str] = []
    seen = set()

    for target in targets:
        cleaned_target = target.strip()
        normalized_target = cleaned_target.lower()

        if not normalized_target:
            continue

        if normalized_target in seen:
            continue

        seen.add(normalized_target)
        unique_targets.append(cleaned_target)

    return unique_targets


def keyword_is_present(
    claim: str,
    keyword: str,
) -> bool:
    """
    Match one complete word or phrase.

    This prevents the keyword "sign" from incorrectly
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


def route_tools(
    claim: str,
) -> ToolRoutingDecision:
    """
    Select tools for one verification claim.

    The Image Inspector is always used.

    OCR is added when the claim depends on visible writing,
    numbers, signs, labels, dates, or other readable content.
    """

    matched_keywords = find_matching_keywords(
        claim
    )

    quoted_targets = find_quoted_text_targets(
        claim
    )

    year_targets = find_year_targets(
        claim
    )

    text_targets = deduplicate_targets(
        quoted_targets + year_targets
    )

    use_ocr = bool(
        matched_keywords
        or text_targets
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