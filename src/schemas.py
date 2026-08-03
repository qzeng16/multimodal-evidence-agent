from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


VerificationLabel = Literal[
    "supported",
    "refuted",
    "insufficient",
]


class VerificationInput(BaseModel):
    """One multimodal claim-verification example."""

    example_id: str
    claim: str
    image_path: str
    context: Optional[str] = None
    gold_label: Optional[VerificationLabel] = None

    # Evaluation annotations
    category: Optional[str] = None
    expected_use_ocr: Optional[bool] = None


class EvidenceItem(BaseModel):
    """One piece of evidence used for verification."""

    modality: Literal[
        "visual",
        "ocr",
        "text",
        "metadata",
    ]

    content: str
    source: str

    relevance: float = Field(
        ge=0.0,
        le=1.0,
    )


class ToolCallRecord(BaseModel):
    """One recorded tool call in the agent workflow."""

    tool_name: str
    tool_input: Dict
    tool_output_summary: str


class VisualInspection(BaseModel):
    """Structured visual evidence extracted from an image."""

    scene_description: str

    supporting_observations: List[str] = Field(
        default_factory=list
    )

    contradicting_observations: List[str] = Field(
        default_factory=list
    )

    visible_text: List[str] = Field(
        default_factory=list
    )

    uncertainty_notes: List[str] = Field(
        default_factory=list
    )


class ToolRoutingDecision(BaseModel):
    """Tools selected for processing one claim."""

    use_image_inspector: bool = True
    use_ocr: bool = False

    reasoning: str

    matched_keywords: List[str] = Field(
        default_factory=list
    )

    text_targets: List[str] = Field(
        default_factory=list
    )


class OCRTextSpan(BaseModel):
    """One text span detected by the OCR tool."""

    text: str
    location: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    relevance_to_claim: float = Field(
        ge=0.0,
        le=1.0,
    )


class OCRExtraction(BaseModel):
    """Structured OCR evidence extracted from an image."""

    detected_text: List[OCRTextSpan] = Field(
        default_factory=list
    )

    target_matches: List[str] = Field(
        default_factory=list
    )

    target_mismatches: List[str] = Field(
        default_factory=list
    )

    uncertainty_notes: List[str] = Field(
        default_factory=list
    )


class VerificationDecision(BaseModel):
    """Intermediate decision produced by the verifier."""

    label: VerificationLabel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str

    relevant_visual_observations: List[str] = Field(
        default_factory=list
    )

    relevant_ocr_observations: List[str] = Field(
        default_factory=list
    )


class VerificationResult(BaseModel):
    """Final structured output from the verification agent."""

    example_id: str
    label: VerificationLabel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str

    routing_decision: ToolRoutingDecision

    evidence: List[EvidenceItem] = Field(
        default_factory=list
    )

    tool_trace: List[ToolCallRecord] = Field(
        default_factory=list
    )