from src.tool_router import route_tools


def test_visual_claim_skips_ocr() -> None:
    """A purely visual claim should not invoke OCR."""

    decision = route_tools(
        "The traffic light in the image is red."
    )

    assert decision.use_image_inspector is True
    assert decision.use_ocr is False
    assert decision.text_targets == []


def test_visible_text_claim_uses_ocr() -> None:
    """A claim about written signage should invoke OCR."""

    decision = route_tools(
        'The street sign says "28th St."'
    )

    assert decision.use_image_inspector is True
    assert decision.use_ocr is True
    assert "28th St." in decision.text_targets


def test_quoted_target_is_extracted() -> None:
    """Quoted text should become an OCR target."""

    decision = route_tools(
        'The bowl says "MADAM MAM\'S".'
    )

    assert decision.use_ocr is True
    assert "MADAM MAM'S" in decision.text_targets


def test_year_target_is_extracted() -> None:
    """A four-digit year should become an OCR target."""

    decision = route_tools(
        "The traffic light was installed in 2024."
    )

    assert decision.use_ocr is True
    assert "2024" in decision.text_targets


def test_router_is_deterministic() -> None:
    """The same claim should always produce the same route."""

    claim = 'The street sign says "28th St."'

    first_decision = route_tools(
        claim
    )

    second_decision = route_tools(
        claim
    )

    assert (
        first_decision.model_dump()
        == second_decision.model_dump()
    )
