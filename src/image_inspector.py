import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.schemas import VisualInspection


DEFAULT_MODEL = "gpt-5.6"


def encode_image_as_data_url(image_path: str) -> str:
    """
    Convert a local image into a Base64 data URL.

    Example output:
        data:image/png;base64,iVBORw0KGgo...
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Image path is not a file: {image_path}"
        )

    mime_type, _ = mimetypes.guess_type(path.name)

    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(
            f"Could not determine image type: {image_path}"
        )

    with path.open("rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


def inspect_image(
    image_path: str,
    claim: str,
    context: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> VisualInspection:
    """
    Analyze an image and extract claim-relevant visual evidence.

    This tool does not make the final verification decision.
    It only reports visible evidence from the image.
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Check that the .env file is saved in the project root."
        )

    image_data_url = encode_image_as_data_url(image_path)

    client = OpenAI(api_key=api_key)

    context_text = context or "No additional context was provided."

    system_prompt = """
You are the Image Inspector tool inside a multimodal evidence
verification agent.

Your task is to inspect the image and extract objective visual
evidence relevant to the supplied claim.

Rules:
1. Use only information that is visibly supported by the image.
2. Do not use outside knowledge.
3. Do not make the final supported, refuted, or insufficient decision.
4. Separate observations that support the claim from observations
   that contradict the claim.
5. Record readable text that appears in the image.
6. Explicitly report ambiguity, poor visibility, occlusion, or other
   uncertainty.
7. Be concise and factual.
""".strip()

    user_prompt = f"""
Claim:
{claim}

Additional context:
{context_text}

Inspect the image and return structured visual evidence relevant
to this claim.
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
                        "detail": "auto",
                    },
                ],
            },
        ],
        text_format=VisualInspection,
    )

    inspection = response.output_parsed

    if inspection is None:
        raise RuntimeError(
            "The model returned no parsed visual inspection."
        )

    return inspection