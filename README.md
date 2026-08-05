# Multimodal Evidence Verification Agent

[![Tests](https://github.com/qzeng16/multimodal-evidence-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/qzeng16/multimodal-evidence-agent/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Testing](https://img.shields.io/badge/tests-38%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-functional%20prototype-orange)

A tool-using multimodal agent that verifies claims against image evidence and returns one of three labels:

- `supported`
- `refuted`
- `insufficient`

The system dynamically routes each claim through visual inspection and optional OCR, extracts structured evidence, performs deterministic text matching, records a complete tool trace, and produces reproducible evaluation artifacts.

---

## Why This Project

A multimodal model can often describe an image, but reliable verification requires more than a plausible description.

A verification system must distinguish between:

- evidence that directly supports a claim;
- evidence that directly contradicts a claim;
- evidence that is missing, unreadable, ambiguous, or irrelevant.

For example, failing to find the text `2024` in an image does not prove that an object was not installed in 2024. The correct result may be `insufficient`, rather than `refuted`.

This project explores an evidence-oriented agent architecture that separates:

1. tool routing;
2. visual perception;
3. blind OCR;
4. deterministic text comparison;
5. final evidence-based reasoning.

---

## Key Features

- Deterministic routing between visual inspection and OCR
- Structured multimodal evidence extraction
- Claim-independent blind OCR
- Multi-view OCR for difficult or rotated text
- Deterministic normalized text matching
- Three-way verification with explicit insufficient-evidence handling
- Complete tool-use traces
- Disk caching for expensive perception calls
- Runtime latency and model-call metrics
- FastAPI service
- Isolated run artifact directories
- Unit tests and GitHub Actions CI

---

## System Architecture

```text
                    Image + Claim
                          |
                          v
                   +--------------+
                   | Tool Router  |
                   +--------------+
                     |          |
             visual  |          | text-dependent
                     v          v
          +----------------+   +----------------------+
          | Image Inspector|   | Multi-View Blind OCR |
          +----------------+   +----------------------+
                     |          |
                     |          v
                     |   +----------------------+
                     |   | Deterministic Matcher|
                     |   +----------------------+
                     |          |
                     +----------+
                          |
                          v
                +----------------------+
                | Verification Reasoner|
                +----------------------+
                          |
                          v
        supported / refuted / insufficient
                          |
                          v
       Evidence + Confidence + Tool Trace + Metrics
```

The final reasoner operates on structured evidence returned by the tools. It does not directly inspect the original image.

---

## Agent Workflow

### 1. Tool Router

The deterministic router analyzes the claim and decides whether OCR is required.

A visual-state claim such as:

```text
The traffic light is red.
```

uses:

```text
tool_router
→ image_inspector
→ verification_reasoner
```

A text-dependent claim such as:

```text
The street sign says "28th St."
```

uses:

```text
tool_router
→ image_inspector
→ ocr_tool
→ verification_reasoner
```

The router detects signals including:

- text-related keywords;
- quoted phrases;
- signs and labels;
- dates and four-digit years;
- license plates;
- written identifiers.

Because the router is deterministic, identical claims always produce identical routing decisions.

---

### 2. Image Inspector

The Image Inspector extracts structured visual evidence:

```json
{
  "scene_description": "A city intersection with red traffic lights.",
  "supporting_observations": [
    "A large blue street sign is visible near the upper-left light."
  ],
  "contradicting_observations": [],
  "visible_text": [
    "28th St",
    "2800 W"
  ],
  "uncertainty_notes": []
}
```

Its responsibilities include:

- describing the relevant scene;
- identifying supporting evidence;
- identifying contradicting evidence;
- reporting visible text informally;
- recording uncertainty and visibility limitations.

---

### 3. Blind OCR

The OCR model does not receive:

- the original claim;
- the expected label;
- the gold answer;
- the target phrase as semantic guidance.

It only receives image views and independently transcribes visible text.

This reduces confirmation bias. The OCR model cannot simply repeat the phrase mentioned in the claim.

After transcription, a deterministic Python matcher compares the OCR result with the target text.

The matcher normalizes:

- capitalization;
- punctuation;
- apostrophe variants;
- repeated whitespace;
- common Unicode variants.

Example:

```text
Target:   28th St.
Detected: 28th St
Result:   Match after normalization
```

---

### 4. Multi-View OCR

Small, curved, rotated, or low-contrast text may be difficult to read from the complete image.

For configured difficult regions, the OCR system can generate:

```text
full_original
region_upscaled
region_rot90
region_rot270
region_high_contrast_rot270
```

The views are processed together in one OCR model call.

The preprocessing pipeline supports:

- configurable normalized crop regions;
- Lanczos upscaling;
- 90-degree rotations;
- grayscale conversion;
- contrast enhancement;
- consolidated multi-view transcription.

OCR region configurations are stored in:

```text
data/ocr_regions.json
```

---

### 5. Deterministic Text Matcher

The deterministic matcher runs after blind OCR.

It produces:

- target matches;
- target mismatches;
- normalized comparisons;
- relevance scores;
- uncertainty notes.

This separation is important:

```text
OCR model:
What text is visible?

Deterministic matcher:
Does that text match the claim target?

Verification reasoner:
What does the combined evidence imply?
```

---

### 6. Verification Reasoner

The Verification Reasoner receives:

- the claim;
- optional context;
- the routing decision;
- structured visual evidence;
- optional OCR evidence;
- deterministic text-match results.

It returns:

```json
{
  "label": "supported",
  "confidence": 0.99,
  "rationale": "The specified sign displays the claimed text.",
  "relevant_visual_observations": [
    "The blue street sign is clearly visible."
  ],
  "relevant_ocr_observations": [
    "OCR detected 28th St with high confidence."
  ]
}
```

For exact-text claims, OCR evidence is preferred over an informal reading from the general Image Inspector.

The reasoner is also instructed not to convert missing evidence into a false contradiction.

Example:

```text
Claim:
The traffic light was installed in 2024.

Observed evidence:
No readable installation date is visible.

Decision:
insufficient
```

---

## Output Labels

### `supported`

The available evidence directly supports the claim.

### `refuted`

The available evidence directly contradicts the claim.

### `insufficient`

The evidence cannot establish whether the claim is true or false.

Common reasons include:

- the relevant detail is not visible;
- the text is unreadable;
- the claim concerns a historical fact;
- identity cannot be established from appearance;
- the image is ambiguous;
- absence of evidence is not evidence of the opposite.

---

## Tool Trace and Observability

Each result includes a complete tool trace.

Example:

```text
1. tool_router
2. image_inspector
3. ocr_tool
4. verification_reasoner
```

Each trace entry records:

- tool name;
- structured tool input;
- concise output summary.

Runtime metrics include:

- routing latency;
- Image Inspector latency;
- OCR latency;
- Verification Reasoner latency;
- total latency;
- cache hits and misses;
- logical model path calls;
- actual model API calls.

A logical model call represents a model-backed stage in the selected tool path.

An actual model API call represents a request that was not served from cache.

---

## Disk Cache

The Image Inspector and OCR outputs can be cached on disk.

Cache keys include:

- image SHA-256;
- tool name;
- tool version;
- relevant tool inputs;
- OCR configuration hash where applicable.

The Verification Reasoner is intentionally executed on every run so that the final decision is generated from the current evidence and reasoning configuration.

Default cache location:

```text
outputs/cache/
├── image_inspector/
└── ocr_tool/
```

Cache files are local runtime artifacts and are not committed to Git.

### Representative Cache Benchmark

The following result came from a local run of `sample_004`.

| Metric | Cached | Cache disabled |
|---|---:|---:|
| Cache hits | 2 | 0 |
| Logical model path calls | 3 | 3 |
| Actual model API calls | 1 | 3 |
| Total latency | 4.943 s | 42.810 s |

For this representative run:

- two model calls were avoided;
- actual model calls decreased from 3 to 1;
- total latency improved by approximately 8.66×;
- total latency decreased by approximately 88.5%.

Latency depends on network conditions and model response time, so these values should be treated as an example rather than a guaranteed benchmark.

---

## Evaluation Dataset

The current functional evaluation set contains:

- 5 images;
- 21 manually designed claims;
- 9 supported claims;
- 7 refuted claims;
- 5 insufficient claims.

The claims exercise:

- direct visual confirmation;
- direct visual contradiction;
- exact text verification;
- punctuation normalization;
- rotated-text recognition;
- visual attributes;
- object recognition;
- spatial relationships;
- visible actions;
- ambiguous evidence;
- non-visible historical facts.

### Evaluation Categories

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

This is a small, curated functional evaluation set. It is intended to test system behavior and failure handling. It is not a generalization benchmark and should not be interpreted as evidence of production-level accuracy.

---

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

Five of the 21 examples require OCR:

```text
5 / 21 = 0.238
```

The router invoked OCR for those five examples.

### Tool-Use Efficiency

| Metric | Result |
|---|---:|
| Average tool calls | 3.238 |
| Average logical model calls | 2.238 |
| Optimal tool-path rate | 1.000 |
| Average extra tool calls | 0.000 |
| Average missing tool calls | 0.000 |

Visual-only claims use:

```text
tool_router
→ image_inspector
→ verification_reasoner
```

Text-dependent claims use:

```text
tool_router
→ image_inspector
→ ocr_tool
→ verification_reasoner
```

The evaluator reports logical tool-path efficiency. Runtime cache metrics separately report actual API calls.

---

## OCR Ablation

A targeted two-example OCR ablation compared single-view blind OCR with multi-view blind OCR.

| Configuration | Verification accuracy | Exact transcription |
|---|---:|---:|
| Blind single-view OCR | 0 / 2 | 0.000 |
| Blind multi-view OCR | 2 / 2 | 1.000 |

The result demonstrates the value of crop, rotation, upscale, and contrast-enhanced views for the selected difficult examples.

This is a small targeted ablation, not a large-scale OCR benchmark.

Detailed outputs are stored in:

```text
experiments/ocr_ablation.json
experiments/ocr_ablation_summary.md
```

---

## Quick Start

### Requirements

- Python 3.9
- An OpenAI API key

### Clone the Repository

```bash
git clone https://github.com/qzeng16/multimodal-evidence-agent.git
cd multimodal-evidence-agent
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Runtime Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Configure the API Key

Create a local `.env` file:

```bash
cat > .env <<'ENVEOF'
OPENAI_API_KEY=your-api-key-here
ENVEOF
```

Do not commit `.env` or expose the API key in logs.

---

## Command-Line Usage

### Run One Example

```bash
python main.py --example-id sample_004
```

### Run One Example Without Cache

```bash
python main.py \
  --example-id sample_004 \
  --no-cache
```

This is useful when measuring uncached latency.

### Run the Full Evaluation Set

```bash
python main.py
```

Running the full set may make multiple model API calls and may incur API costs.

### View CLI Help

```bash
python main.py --help
```

---

## FastAPI Service

Start the development server:

```bash
python -m uvicorn app:app --reload
```

Available documentation:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

### Health Check

```bash
curl -s \
  http://127.0.0.1:8000/health \
  | python -m json.tool
```

### Verify a Dataset Example

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/verify-example/sample_004?use_cache=true" \
  | python -m json.tool
```

### Verify a Custom Claim

The image must be located inside `data/images`.

```bash
curl -s -X POST \
  http://127.0.0.1:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/images/sample_001.png",
    "claim": "The street sign says \"28th St.\"",
    "context": "The claim refers to the large blue street sign near the upper-left traffic light.",
    "use_cache": true
  }' \
  | python -m json.tool
```

A response includes:

```json
{
  "example_id": "api_request",
  "label": "supported",
  "confidence": 0.99,
  "rationale": "The sign displays the claimed text.",
  "cache_enabled": true,
  "cache_hits": 2,
  "cache_misses": 0,
  "cache_hit_rate": 1.0,
  "logical_model_call_count": 3,
  "actual_model_call_count": 1
}
```

The complete response also contains selected evidence, routing information, the tool trace, and latency metrics.

---

## Run Artifacts

Each CLI execution creates an isolated directory under:

```text
outputs/runs/
```

Example:

```text
outputs/runs/
└── 20260805T014254Z_sample_004_6c53945b/
    ├── predictions.jsonl
    ├── metrics.json
    └── run_manifest.json
```

### `predictions.jsonl`

Contains one record per evaluated example, including:

- claim;
- predicted label;
- confidence;
- rationale;
- selected evidence;
- routing decision;
- tool trace.

### `metrics.json`

Contains dataset-level metrics for:

- classification;
- per-label accuracy;
- per-category accuracy;
- confusion matrix;
- OCR routing;
- tool-use efficiency.

### `run_manifest.json`

Records execution metadata such as:

```json
{
  "run_id": "20260805T014254Z_sample_004_6c53945b",
  "metadata": {
    "dataset_path": "data/samples.jsonl",
    "example_id_filter": "sample_004",
    "cache_enabled": true,
    "evaluated_examples": 1,
    "accuracy": 1.0,
    "runtime": {
      "execution_count": 1,
      "cache_hits": 2,
      "cache_misses": 0,
      "logical_model_calls": 3,
      "actual_model_calls": 1,
      "model_calls_avoided": 2
    }
  }
}
```

Run directories are local experiment artifacts and are ignored by Git.

---

## Testing

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the complete test suite:

```bash
python -m pytest
```

Run tests with detailed names:

```bash
python -m pytest -v
```

The current suite contains 38 tests covering:

- dataset loading and validation;
- example selection;
- deterministic tool routing;
- OCR target extraction;
- cache key stability;
- cache read and write behavior;
- corrupted cache recovery;
- file hashing;
- API route registration;
- health metadata;
- image path security;
- run directory creation;
- JSON and JSONL artifact writing;
- manifest generation.

The tests do not call the OpenAI API.

---

## Continuous Integration

GitHub Actions runs the test suite on:

- pushes to `main`;
- pull requests.

Workflow file:

```text
.github/workflows/tests.yml
```

The CI environment:

1. checks out the repository;
2. installs Python 3.9;
3. installs development dependencies;
4. runs `python -m pytest`.

---

## Project Structure

```text
multimodal-evidence-agent/
├── app.py
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── README.md
│
├── data/
│   ├── images/
│   ├── ocr_regions.json
│   └── samples.jsonl
│
├── experiments/
│   ├── analyze_ablation.py
│   ├── ocr_ablation.json
│   └── ocr_ablation_summary.md
│
├── src/
│   ├── cache.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── image_inspector.py
│   ├── image_loader.py
│   ├── ocr_tool.py
│   ├── pipeline.py
│   ├── run_artifacts.py
│   ├── schemas.py
│   ├── tool_router.py
│   └── verifier.py
│
├── tests/
│   ├── test_app.py
│   ├── test_cache.py
│   ├── test_dataset.py
│   ├── test_run_artifacts.py
│   └── test_tool_router.py
│
├── outputs/
│   ├── cache/
│   └── runs/
│
└── .github/
    └── workflows/
        └── tests.yml
```

---

## Security and Data Handling

The API restricts local image access to files inside:

```text
data/images/
```

This prevents API requests from reading arbitrary local files.

Additional safeguards:

- API keys are loaded from environment variables;
- `.env` is excluded from Git;
- runtime cache files are excluded from Git;
- run artifacts are excluded from Git;
- custom remote image URLs are not fetched by the API.

This project is a research and portfolio prototype and has not undergone a production security audit.

---

## Current Limitations

- The evaluation set is small and manually curated.
- Difficult OCR regions currently rely on configured crops.
- Performance depends on the underlying multimodal model.
- Confidence values are model-generated and are not formally calibrated.
- The cache uses the local filesystem rather than a distributed store.
- The system does not retrieve external textual evidence.
- The API processes local project images rather than uploaded files.
- There is no browser-based user interface.
- The current evaluation does not measure robustness to large distribution shifts.

---

## Possible Extensions

- Larger and more diverse evaluation datasets
- Automatic text-region detection
- Learned or model-assisted tool routing
- Confidence calibration
- Adversarial and counterfactual test cases
- Uploaded-image API support
- Async batch processing
- Docker packaging
- Cloud deployment
- Human review for low-confidence outputs
- External evidence retrieval
- Multimodal retrieval-augmented verification

---

## Design Principles

The implementation follows five core principles:

1. **Route tools intentionally.**  
   Expensive tools should only be used when they are relevant.

2. **Separate perception from verification.**  
   A perception model should describe evidence rather than decide the final answer immediately.

3. **Reduce confirmation bias.**  
   OCR should transcribe independently before target matching occurs.

4. **Treat uncertainty as a valid result.**  
   Missing or ambiguous evidence should not be forced into support or contradiction.

5. **Make experiments reproducible.**  
   Tool traces, metrics, cache statistics, and run manifests should be preserved.

---

## Repository

```text
https://github.com/qzeng16/multimodal-evidence-agent
```
