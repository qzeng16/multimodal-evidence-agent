# Multimodal Evidence Verification Agent

A tool-using multimodal agent that verifies claims against image evidence and returns one of three labels:

- `supported`
- `refuted`
- `insufficient`

The system dynamically selects visual and OCR tools, records its tool-use trace, extracts structured evidence, and evaluates both final verification accuracy and agent routing efficiency.

## Overview

Multimodal models can often describe an image, but reliable claim verification requires more than producing a plausible answer.

A verification system should distinguish between:

- evidence that directly supports a claim;
- evidence that directly contradicts a claim;
- evidence that is missing, ambiguous, unreadable, or insufficient.

This project implements an evidence-oriented agent that separates perception, OCR, deterministic text comparison, and final reasoning into distinct stages.

The system is designed around four principles:

1. Use explicit evidence instead of unsupported assumptions.
2. Dynamically invoke OCR only when the claim depends on visible text.
3. Keep OCR independent from the target claim to reduce confirmation bias.
4. Return `insufficient` when the image cannot establish the requested fact.

## System Architecture

```text
Image + Claim
      |
      v
Tool Router
      |
      +------------------------------+
      |                              |
      v                              v
Image Inspector              Multi-View Blind OCR
                                     |
                                     v
                           Deterministic Text Matcher
      |                              |
      +---------------+--------------+
                      |
                      v
            Verification Reasoner
                      |
                      v
     supported / refuted / insufficient
                      |
                      v
       Evidence + Confidence + Tool Trace
```

### 1. Tool Router

The router analyzes the claim and decides whether OCR is necessary.

Visual claims such as:

```text
The traffic light is red.
```

use:

```text
Tool Router
→ Image Inspector
→ Verification Reasoner
```

Text-dependent claims such as:

```text
The street sign says "28th St."
```

use:

```text
Tool Router
→ Image Inspector
→ OCR Tool
→ Verification Reasoner
```

The router detects:

- text-related keywords;
- quoted text targets;
- street names and labels;
- dates and four-digit years;
- license plates and other written identifiers.

### 2. Image Inspector

The Image Inspector extracts structured visual evidence:

- scene description;
- supporting observations;
- contradicting observations;
- visible text;
- uncertainty notes.

The final verifier does not inspect the original image directly. It reasons over structured evidence produced by the selected tools.

### 3. Blind OCR

The OCR model does not receive:

- the original claim;
- the target phrase;
- the expected label;
- the gold answer.

It only receives image views and independently transcribes visible text.

This prevents the OCR model from simply reproducing the phrase mentioned in the claim.

After OCR finishes, a deterministic Python matcher compares the transcription with the target phrase.

The matcher normalizes:

- capitalization;
- punctuation;
- apostrophe variations;
- repeated whitespace;
- common Unicode variations.

For example:

```text
Target:   28th St.
Detected: 28th St
Result:   Match after normalization
```

### 4. Multi-View OCR

Small, curved, rotated, or low-contrast text can be difficult to read from the complete image.

For selected difficult regions, the OCR tool generates several views:

```text
full_original
region_upscaled
region_rot90
region_rot270
region_high_contrast_rot270
```

All views are processed in one OCR model call.

The current preprocessing pipeline supports:

- manually configured OCR regions;
- cropping;
- Lanczos upscaling;
- 90-degree rotations;
- grayscale conversion;
- automatic contrast enhancement;
- multi-view transcription consolidation.

### 5. Verification Reasoner

The Verification Reasoner receives:

- the claim;
- optional context;
- the routing decision;
- structured visual evidence;
- optional OCR evidence;
- deterministic target-match results.

It returns:

```json
{
  "label": "supported",
  "confidence": 0.99,
  "rationale": "The visible evidence directly supports the claim.",
  "relevant_visual_observations": [],
  "relevant_ocr_observations": []
}
```

For exact-text claims, high-quality OCR evidence is preferred over informal text readings from the general Image Inspector.

The verifier is instructed not to treat missing text as automatic contradiction. For example:

```text
Claim:
The traffic light was installed in 2024.

OCR result:
The text "2024" was not detected.

Correct decision:
insufficient
```

The absence of a visible date does not prove that the light was not installed in 2024.

## Agent Output

Each result contains:

- final label;
- confidence;
- concise rationale;
- selected evidence;
- routing decision;
- complete tool trace.

Example tool trace:

```text
1. tool_router
2. image_inspector
3. ocr_tool
4. verification_reasoner
```

## Evaluation Dataset

The current evaluation set contains:

- 5 images;
- 21 manually designed claims;
- 9 supported claims;
- 7 refuted claims;
- 5 insufficient claims.

The images cover street scenes, food, utensils, printed text, rotated text, spatial relations, and outdoor actions.

### Categories

| Category | Examples |
|---|---:|
| visual_state | 2 |
| non_visible_fact | 4 |
| visible_text | 2 |
| spatial_relation | 2 |
| visual_attribute | 3 |
| visual_ambiguity | 1 |
| visual_content | 2 |
| visual_object | 2 |
| rotated_text | 2 |
| visual_action | 1 |
| **Total** | **21** |

The dataset includes claims requiring:

- direct visual confirmation;
- visual contradiction;
- exact text verification;
- OCR normalization;
- rotated-text recognition;
- spatial reasoning;
- recognition of insufficient evidence;
- rejection of unsupported identity or historical assumptions.

## Evaluation Results

### Verification Performance

| Metric | Result |
|---|---:|
| Total examples | 21 |
| Correct examples | 21 |
| Failed examples | 0 |
| Accuracy | 1.000 |
| Average confidence | 0.970 |

### Per-Label Accuracy

| Gold label | Correct | Accuracy |
|---|---:|---:|
| supported | 9 / 9 | 1.000 |
| refuted | 7 / 7 | 1.000 |
| insufficient | 5 / 5 | 1.000 |

### Tool Routing

| Metric | Result |
|---|---:|
| Router accuracy | 1.000 |
| OCR precision | 1.000 |
| OCR recall | 1.000 |
| OCR F1 | 1.000 |
| OCR invocation rate | 0.238 |
| Unnecessary OCR rate | 0.000 |
| Missed OCR rate | 0.000 |

Five of the 21 claims required OCR:

```text
5 / 21 = 0.238
```

The router invoked OCR for exactly those five examples.

### Tool-Use Efficiency

| Metric | Result |
|---|---:|
| Average tool calls | 3.238 |
| Average model calls | 2.238 |
| Optimal tool-path rate | 1.000 |
| Average extra tool calls | 0.000 |
| Average missing tool calls | 0.000 |

For 16 visual-only claims, the system used three tools:

```text
tool_router
image_inspector
verification_reasoner
```

For five text-dependent claims, it used four tools:

```text
tool_router
image_inspector
ocr_tool
verification_reasoner
```

Therefore:

```text
Average tool calls
= (16 × 3 + 5 × 4) / 21
= 3.238
```

The router itself is deterministic and does not require a model call:

```text
Average model calls
= (16 × 2 + 5 × 3) / 21
= 2.238
```

## OCR Failure Analysis

A difficult example contained small, curved, rotated text printed on the rim of a bowl.

The ground-truth text was:

```text
MADAM MAM'S
```

### Claim-Conditioned OCR Failure

An early OCR implementation received both the image and the target phrase.

On the same physical text region, different runs produced inconsistent outputs such as:

```text
SIAM VILLAGE
SIAM I WOK
```

This exposed a potential target-leakage failure:

> When the OCR model knows the phrase being tested, ambiguous visual evidence can be completed toward that phrase.

These early runs are treated as qualitative observations rather than formal ablation results because the original annotation was later corrected.

### Blind Single-View OCR

The OCR model was then separated from the claim and received only the original image.

It produced:

```text
JIMMY WONG'S
SIAM INN TOO
```

The system correctly recognized that the evidence was ambiguous and returned `insufficient`, but both formal verification examples were incorrect.

### Multi-View Blind OCR

The final method used:

- the complete image;
- an enlarged crop;
- a 90-degree rotation;
- a 270-degree rotation;
- a high-contrast rotated crop.

It independently transcribed:

```text
MADAM MAM'S
```

under both the supported and refuted claims.

## OCR Ablation Experiment

| Method | Views | Verification Accuracy | OCR Transcription Accuracy | Consistent Across Claims | Average OCR Confidence |
|---|---:|---:|---:|---:|---:|
| Blind Single-View OCR | 1 | 0.000 | 0.000 | No | 0.785 |
| Multi-View Blind OCR | 5 | 1.000 | 1.000 | Yes | 0.950 |

Average verification confidence also increased:

```text
0.790 → 0.975
```

The complete experiment report is available at:

```text
experiments/ocr_ablation_summary.md
```

The structured experiment data is stored at:

```text
experiments/ocr_ablation.json
```

Run the analysis with:

```bash
python experiments/analyze_ablation.py
```

## Evidence Ambiguity Example

One original claim stated:

```text
The food is topped with orange fish and a pale sauce.
```

The image clearly established:

- an orange topping;
- a pale sauce.

However, the image alone could not strictly establish that the orange topping was fish.

The verifier returned:

```text
insufficient
```

The annotation was revised from `supported` to `insufficient`.

This example demonstrates the intended evidence standard:

> Visual resemblance should not automatically be treated as confirmed identity.

## Repository Structure

```text
multimodal-evidence-agent/
├── main.py
├── test_router.py
├── requirements.txt
├── download_selected_openimages.py
├── data/
│   ├── samples.jsonl
│   ├── ocr_regions.json
│   └── images/
│       ├── sample_001.png
│       └── openimages/
├── experiments/
│   ├── analyze_ablation.py
│   ├── ocr_ablation.json
│   └── ocr_ablation_summary.md
├── outputs/
│   ├── predictions.jsonl
│   └── metrics.json
└── src/
    ├── schemas.py
    ├── image_loader.py
    ├── image_inspector.py
    ├── tool_router.py
    ├── ocr_tool.py
    ├── verifier.py
    └── evaluator.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git.

## Usage

### Run the complete evaluation set

```bash
python main.py
```

### Run one example

```bash
python main.py --example-id sample_012
```

### Show command-line help

```bash
python main.py --help
```

### Test the deterministic router

```bash
python test_router.py
```

### Run the OCR ablation analysis

```bash
python experiments/analyze_ablation.py
```

## Saved Outputs

Predictions are written to:

```text
outputs/predictions.jsonl
```

Metrics are written to:

```text
outputs/metrics.json
```

Each prediction record includes:

- claim and image information;
- gold and predicted labels;
- confidence;
- route correctness;
- OCR decision;
- tool names;
- tool-call count;
- model-call count;
- selected evidence;
- complete tool trace.

Note: running a single example currently overwrites the saved prediction and metrics files with that single-example result.

## Design Decisions

### Why Use Three Labels?

Binary supported/refuted classification encourages systems to guess when evidence is missing.

The `insufficient` label explicitly represents:

- unreadable text;
- ambiguous object identity;
- unsupported dates;
- unknown locations;
- unknown preparation time;
- historical facts not visible in the image;
- conflicts between uncertain perception tools.

### Why Use Deterministic Text Matching?

A language model should not decide whether:

```text
28th St.
```

matches:

```text
28th St
```

A deterministic matcher makes this comparison:

- reproducible;
- inspectable;
- independent of model variation;
- easier to test.

### Why Keep OCR Blind?

Claim-conditioned OCR can create confirmation bias.

Blind OCR separates:

```text
What text is visible?
```

from:

```text
Does that text support the claim?
```

This separation produces a clearer and more auditable agent pipeline.

## Limitations

The current results should be interpreted as a functional evaluation, not as proof of broad generalization.

Current limitations include:

- only five images;
- 21 manually curated claims;
- one focused rotated-text ablation;
- manually configured crop coordinates for the difficult OCR region;
- no large-scale public benchmark evaluation;
- no automated text-region detector;
- no repeated-run robustness statistics;
- no latency or API-cost evaluation;
- model outputs may vary across repeated executions;
- the general Image Inspector still receives the claim and can exhibit confirmation bias.

The 1.000 evaluation accuracy therefore describes this small curated dataset only.

## Future Work

Planned extensions include:

1. Automatic text-region proposal and crop selection.
2. Multiple OCR regions per image.
3. Repeated-run stability evaluation.
4. Confidence calibration.
5. Latency and API-cost tracking.
6. OCR caching across claims sharing the same image.
7. Larger public multimodal verification benchmarks.
8. Automatic selection between original, rotated, and enhanced views.
9. Additional tools for metadata and external evidence retrieval.
10. A lightweight API or web interface.

## Main Takeaway

This project demonstrates that reliable multimodal verification requires more than asking one model to inspect an image and answer a question.

A more robust pipeline separates:

```text
routing
perception
OCR
deterministic comparison
verification
evaluation
```

The rotated-text failure case shows how claim-conditioned perception can become unstable, while multi-view blind OCR and deterministic matching provide a more reliable and auditable alternative.