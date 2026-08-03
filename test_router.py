from typing import List, Tuple

from src.tool_router import route_tools


TestCase = Tuple[
    str,
    bool,
    List[str],
]


TEST_CASES: List[TestCase] = [
    (
        "The traffic light in the image is red.",
        False,
        [],
    ),
    (
        'The street sign says "28th St."',
        True,
        ["28th St."],
    ),
    (
        "The license plate reads 8EVX265.",
        True,
        [],
    ),
    (
        "There are two cars in the image.",
        False,
        [],
    ),
    (
        "The traffic light was installed in 2024.",
        True,
        ["2024"],
    ),
]


def main() -> None:
    """Test deterministic tool-routing behavior."""

    passed = 0

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        (
            claim,
            expected_use_ocr,
            expected_targets,
        ) = test_case

        decision = route_tools(
            claim
        )

        print(
            f"Test {index}: {claim}"
        )

        print(
            f"Use OCR: "
            f"{decision.use_ocr}"
        )

        print(
            f"Text targets: "
            f"{decision.text_targets}"
        )

        use_ocr_correct = (
            decision.use_ocr
            == expected_use_ocr
        )

        targets_correct = (
            decision.text_targets
            == expected_targets
        )

        print(
            f"Use OCR correct: "
            f"{use_ocr_correct}"
        )

        print(
            f"Targets correct: "
            f"{targets_correct}"
        )

        print("-" * 70)

        if not use_ocr_correct:
            raise AssertionError(
                f"Unexpected OCR decision for: {claim}"
            )

        if not targets_correct:
            raise AssertionError(
                f"Unexpected text targets for: {claim}"
            )

        passed += 1

    print(
        f"Router tests passed: "
        f"{passed}/{len(TEST_CASES)}"
    )


if __name__ == "__main__":
    main()