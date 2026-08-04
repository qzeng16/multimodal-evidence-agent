# OCR Ablation Experiment

Comparison of single-view blind OCR and multi-view blind OCR on small, curved, and rotated text printed on a bowl rim.

## Experimental Setup

- Image: `data/images/openimages/583a3460c32053a1.jpg`
- Ground-truth text: `MADAM MAM'S`
- Evaluation examples: 2

## Results

| Method | Views | Verification Accuracy | OCR Transcription Accuracy | Consistent Across Claims | Average OCR Confidence |
|---|---:|---:|---:|---:|---:|
| Blind Single-View OCR | 1 | 0.000 | 0.000 | No | 0.785 |
| Multi-View Blind OCR | 5 | 1.000 | 1.000 | Yes | 0.950 |

## Per-Example Results

### Blind Single-View OCR

- `sample_012`: OCR=`JIMMY WONG'S`, gold=`supported`, prediction=`insufficient`, correct=`False`
- `sample_013`: OCR=`SIAM INN TOO`, gold=`refuted`, prediction=`insufficient`, correct=`False`

### Multi-View Blind OCR

- `sample_012`: OCR=`MADAM MAM'S`, gold=`supported`, prediction=`supported`, correct=`True`
- `sample_013`: OCR=`MADAM MAM'S`, gold=`refuted`, prediction=`refuted`, correct=`True`

## Qualitative Failure Observation

The earlier claim-conditioned OCR runs are excluded from formal accuracy because they used an initial annotation that was later corrected.

They are retained as evidence that directly providing a target phrase to perception can create instability and potential confirmation bias.

- Run 1: target=`SIAM VILLAGE`, OCR=`SIAM VILLAGE`, prediction=`supported`
- Run 2: target=`SIAM VILLAGE`, OCR=`SIAM I WOK`, prediction=`refuted`

## Main Finding

Blind single-view OCR failed on both rotated-text examples and produced inconsistent transcriptions for the same physical text.

Multi-view blind OCR correctly transcribed `MADAM MAM'S` under both claims and improved verification accuracy from 0/2 to 2/2 without exposing the OCR model to the claim target.

## Limitations

This is a focused two-example ablation on one difficult image. It demonstrates the mechanism and failure mode but is not a large-scale OCR benchmark.
