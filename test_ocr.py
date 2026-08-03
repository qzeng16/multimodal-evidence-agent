from src.ocr_tool import extract_text_from_image
from src.tool_router import route_tools


IMAGE_PATH = "data/images/sample_001.png"

CLAIM = 'The street sign says "28th St."'


def main() -> None:
    """Test the OCR tool with one text-related claim."""

    print(f"Image: {IMAGE_PATH}")
    print(f"Claim: {CLAIM}")

    routing_decision = route_tools(
        CLAIM
    )

    print("\nRouting decision:")
    print(
        "Use Image Inspector: "
        f"{routing_decision.use_image_inspector}"
    )
    print(
        "Use OCR: "
        f"{routing_decision.use_ocr}"
    )
    print(
        "Text targets: "
        f"{routing_decision.text_targets}"
    )

    if not routing_decision.use_ocr:
        print(
            "\nOCR was not selected. "
            "The test will stop."
        )
        return

    print("\nCalling OCR Tool...")

    extraction = extract_text_from_image(
        image_path=IMAGE_PATH,
        claim=CLAIM,
        text_targets=(
            routing_decision.text_targets
        ),
    )

    print("\nOCR extraction:")

    if extraction.detected_text:
        for index, text_span in enumerate(
            extraction.detected_text,
            start=1,
        ):
            print(
                f"{index}. Text: "
                f"{text_span.text}"
            )
            print(
                f"   Location: "
                f"{text_span.location}"
            )
            print(
                f"   Confidence: "
                f"{text_span.confidence:.2f}"
            )
            print(
                f"   Relevance: "
                f"{text_span.relevance_to_claim:.2f}"
            )
    else:
        print("- No readable text detected")

    print("\nTarget matches:")

    if extraction.target_matches:
        for match in extraction.target_matches:
            print(f"- {match}")
    else:
        print("- None")

    print("\nTarget mismatches:")

    if extraction.target_mismatches:
        for mismatch in (
            extraction.target_mismatches
        ):
            print(f"- {mismatch}")
    else:
        print("- None")

    print("\nUncertainty notes:")

    if extraction.uncertainty_notes:
        for note in (
            extraction.uncertainty_notes
        ):
            print(f"- {note}")
    else:
        print("- None")


if __name__ == "__main__":
    main()