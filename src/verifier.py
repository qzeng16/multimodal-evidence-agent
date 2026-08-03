import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.image_inspector import DEFAULT_MODEL
from src.schemas import (
    EvidenceItem,
    OCRExtraction,
    ToolCallRecord,
    ToolRoutingDecision,
    VerificationDecision,
    VerificationInput,
    VerificationResult,
    VisualInspection,
)


def verify_claim(
    example: VerificationInput,
    inspection: VisualInspection,
    routing_decision: ToolRoutingDecision,
    ocr_extraction: Optional[OCRExtraction] = None,
    model: str = DEFAULT_MODEL,
) -> VerificationResult:
    """
    Verify a claim using visual evidence and optional OCR evidence.

    The verifier does not inspect the original image directly.
    It reasons only over evidence produced by selected tools.
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

    client = OpenAI(
        api_key=api_key
    )

    context_text = (
        example.context
        if example.context
        else "No additional context was provided."
    )

    routing_json = routing_decision.model_dump_json(
        indent=2
    )

    visual_evidence_json = inspection.model_dump_json(
        indent=2
    )

    if ocr_extraction is not None:
        ocr_evidence_text = (
            ocr_extraction.model_dump_json(
                indent=2
            )
        )
    else:
        ocr_evidence_text = (
            "The OCR tool was not selected for this claim."
        )

    system_prompt = """
You are the Verification Reasoner inside a multimodal
evidence verification agent.

You receive:

1. A textual claim.
2. Optional context.
3. A tool-routing decision.
4. Structured evidence from an Image Inspector.
5. Optional structured evidence from an OCR Tool.

Classify the claim as:

- supported
- refuted
- insufficient

Definitions:

SUPPORTED:
The available evidence directly and clearly establishes
the claim.

REFUTED:
The available evidence directly and clearly contradicts
the claim.

INSUFFICIENT:
The evidence is absent, ambiguous, incomplete, unreadable,
or does not establish the claim strongly enough.

Rules:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not assume facts absent from the evidence.
4. Absence of evidence is not automatically contradiction.
5. For exact-text claims, prefer OCR evidence over informal
   text readings from the Image Inspector.
6. Ignore ordinary sentence punctuation differences when
   comparing visible text.
7. If OCR detects the requested text after normalization,
   that supports the exact-text claim.
8. If OCR clearly detects a different value at the specified
   target location, that can directly refute the claim.
9. A target not being found is not by itself sufficient for
   refutation when the relevant region is unreadable.
10. A visible object's design or apparent condition cannot
    establish an installation date.
11. Use insufficient for non-visible historical facts,
    causes, ownership, dates, or identities that cannot be
    established from the supplied evidence.
12. Confidence must reflect evidence quality and ambiguity.
13. Put visual evidence only in
    relevant_visual_observations.
14. Put OCR-derived evidence only in
    relevant_ocr_observations.
15. Keep the rationale concise and factual.
""".strip()

    user_prompt = f"""
Claim:
{example.claim}

Additional context:
{context_text}

Tool-routing decision:
{routing_json}

Image Inspector evidence:
{visual_evidence_json}

OCR evidence:
{ocr_evidence_text}

Return the final structured verification decision.
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
                "content": user_prompt,
            },
        ],
        text_format=VerificationDecision,
    )

    decision = response.output_parsed

    if decision is None:
        raise RuntimeError(
            "The model returned no parsed verification decision."
        )

    evidence_items: List[EvidenceItem] = []

    for observation in (
        decision.relevant_visual_observations
    ):
        evidence_items.append(
            EvidenceItem(
                modality="visual",
                content=observation,
                source="image_inspector",
                relevance=1.0,
            )
        )

    for observation in (
        decision.relevant_ocr_observations
    ):
        evidence_items.append(
            EvidenceItem(
                modality="ocr",
                content=observation,
                source="ocr_tool",
                relevance=1.0,
            )
        )

    tool_trace: List[ToolCallRecord] = []

    tool_trace.append(
        ToolCallRecord(
            tool_name="tool_router",
            tool_input={
                "claim": example.claim,
            },
            tool_output_summary=(
                "use_image_inspector="
                f"{routing_decision.use_image_inspector}; "
                "use_ocr="
                f"{routing_decision.use_ocr}; "
                f"{routing_decision.reasoning}"
            ),
        )
    )

    tool_trace.append(
        ToolCallRecord(
            tool_name="image_inspector",
            tool_input={
                "image_path": example.image_path,
                "claim": example.claim,
                "context": example.context,
            },
            tool_output_summary=(
                inspection.scene_description
            ),
        )
    )

    if ocr_extraction is not None:
        tool_trace.append(
            ToolCallRecord(
                tool_name="ocr_tool",
                tool_input={
                    "image_path": example.image_path,
                    "claim": example.claim,
                    "text_targets": (
                        routing_decision.text_targets
                    ),
                },
                tool_output_summary=(
                    f"Detected "
                    f"{len(ocr_extraction.detected_text)} "
                    f"text span(s); "
                    f"{len(ocr_extraction.target_matches)} "
                    f"target match(es); "
                    f"{len(ocr_extraction.target_mismatches)} "
                    f"target mismatch(es)."
                ),
            )
        )

    tool_trace.append(
        ToolCallRecord(
            tool_name="verification_reasoner",
            tool_input={
                "claim": example.claim,
                "used_ocr": (
                    ocr_extraction is not None
                ),
                "visual_observation_count": len(
                    decision.relevant_visual_observations
                ),
                "ocr_observation_count": len(
                    decision.relevant_ocr_observations
                ),
            },
            tool_output_summary=(
                f"Label: {decision.label}; "
                f"confidence: "
                f"{decision.confidence:.2f}"
            ),
        )
    )

    return VerificationResult(
        example_id=example.example_id,
        label=decision.label,
        confidence=decision.confidence,
        rationale=decision.rationale,
        routing_decision=routing_decision,
        evidence=evidence_items,
        tool_trace=tool_trace,
    )