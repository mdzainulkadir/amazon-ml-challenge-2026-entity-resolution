# Business Entity Resolution — Team Wolf Crew

A two-stage entity resolution pipeline: multi-key adaptive blocking (candidate
generation) followed by a LightGBM pairwise classifier, threshold selection,
and one-to-one assignment. See `../../Documentation_template.md` for the full
methodology writeup.

## Directory layout

```
business_entity_resolution/
├── src/
│   ├── normalize.py    # text normalization (transliteration, abbreviations, skeleton keys)
│   ├── blocking.py      # adaptive per-country blocking + residual pass for weak entities
│   ├── features.py      # ~24-column pairwise feature builder (fixed column order)
│   ├── train.py         # trains the model, searches the decision threshold
│   └── predict.py       # end-to-end inference -> the two required output files
├── README.md            # this file
└── requirements.txt     # pinned dependencies
```

## Expected data layout

Run from a working directory containing:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

(This is the same `dataset/` layout the challenge's own
`utils/validate_submission.py` expects.)

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Reproduce end-to-end

From the `business_entity_resolution/` directory, with `dataset/` present
alongside it (adjust `--data-dir` if your data lives elsewhere):

```bash
# 1. Train the model and select the decision threshold.
#    Writes artifacts/model.pkl, artifacts/threshold.json, artifacts/feature_names.json
python3 src/train.py --data-dir dataset --out-dir artifacts

# 2. Run inference on the test set.
#    Writes output/matching_results.tsv and output/candidate_pairs.tsv
python3 src/predict.py --data-dir dataset --artifacts-dir artifacts --out-dir output
```

Then validate the output before submitting anywhere:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Notes on reproducibility

- `train.py`'s LightGBM calls do not currently pin a `random_state` on the
  classifier or `GroupKFold`; re-running training will therefore produce a
  model with the same architecture and very similar, but not bit-identical,
  performance each time. The reported out-of-fold F0.5 and chosen threshold
  in `Documentation_template.md` come from one such run; expect run-to-run
  variation in the third decimal place, not a qualitative change.
- `train.py` and `predict.py` both call `adaptive_block` with the same
  per-country `TARGET_MAXB` starting points; the function itself will back
  off automatically (and print each attempt) if a target proves too large
  for the available memory/compute budget on your machine — this is
  expected behaviour, not an error.
- All normalization, blocking, and feature logic is pure Python / pandas /
  numpy / rapidfuzz — no GPU is required for any part of this pipeline.
- No external data, API, or lookup service is used anywhere in this code.

## Model license and size

The trained classifier is **LightGBM** (MIT License), configured with 500
trees — several orders of magnitude under the challenge's 8-billion-parameter
ceiling.
