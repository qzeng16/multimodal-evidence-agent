from typing import List

from src.tool_router import route_tools


TEST_CLAIMS: List[str] = [
    "The traffic light in the image is red.",
    "The street sign says 28th St.",
    'The street sign says "28th St."',
    "The license plate reads 8EVX265.",
    "There are two cars in the image.",
    "The traffic light was installed in 2024.",
]


def print_routing_result(
    claim: str,
) -> None:
    """Run the router and print one decision."""

    decision = route_tools(
        claim
    )

    print(f"Claim: {claim}")

    print(
        "Use Image Inspector: "
        f"{decision.use_image_inspector}"
    )

    print(
        "Use OCR: "
        f"{decision.use_ocr}"
    )

    print(
        "Reasoning: "
        f"{decision.reasoning}"
    )

    print(
        "Matched keywords: "
        f"{decision.matched_keywords}"
    )

    print(
        "Text targets: "
        f"{decision.text_targets}"
    )

    print("-" * 70)


def main() -> None:
    """Test the router with several types of claims."""

    print(
        f"Testing {len(TEST_CLAIMS)} claim(s).\n"
    )

    for claim in TEST_CLAIMS:
        print_routing_result(
            claim
        )


if __name__ == "__main__":
    main()