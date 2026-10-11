# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end entity resolution pipeline built for Amazon's national 72-hour ML hackathon (2026 edition), matching business records across three independent, noisy data sources with no shared identifiers.

**Team Wolf Crew** — Md Zainul Kadir & Sameer Ali
Gyan Ganga College of Technology (GGCT), Jabalpur

---

## The problem

Given three data sources of business listings (name + address only, no shared IDs, frequent typos/abbreviations/transliteration), find every record in Source 2 and Source 3 that refers to the same real-world business as a given Source 1 entity. A Source 1 entity may match zero, one, or many records — and the test set includes a country (France) that never appears anywhere in training.

Full problem statement and methodology: see [`Documentation_template.md`](./Documentation_template.md).

## Results

| Metric | Value |
|---|---|
| Scale | 1.7M+ Source 1 records matched against a pool of ~10M Source 2/3 records |
| Macro F0.5 (out-of-fold, training data) | ≈ 0.967 |
| Public leaderboard score | **0.9213** (improved from an initial 0.9139 through iterative blocking/feature work) |
| Initial live-leaderboard rank | 992nd of teams competing, out of 89,000+ registered participants nationwide |
| Blocking recall (US / India) | ≈ 94.4% / 88.1% |

## Approach

A two-stage pipeline, built with no external APIs, no geocoding services, and no pre-built entity-resolution libraries — every normalization rule, blocking key, and feature is hand-engineered.

1. **Candidate generation (blocking)** — per-country, multi-key blocking (rare-token, prefix, consonant-skeleton, and address-number keys) with a cap on key frequency that is *estimated and auto-tuned* before the expensive join runs, so the pipeline adapts to whatever compute is actually available instead of relying on a fixed, manually-guessed constant. A second, targeted pass gives extra retrieval effort only to entities poorly served by the first pass.
2. **Matching** — a LightGBM classifier scores every candidate pair on ~24 engineered features (string similarity, IDF-weighted token overlap, and "competition" features that let the model learn one-to-one assignment behaviour on its own), trained with `GroupKFold` cross-validation to prevent any entity's candidates from leaking across folds.
3. **Decision** — a probability threshold chosen by directly optimizing the competition's own macro-F0.5 metric, followed by a one-to-one assignment rule (since no Source 2/3 record was ever a true match for more than one Source 1 entity in the training data).

Full technical write-up, feature list, and error analysis: [`Documentation_template.md`](./Documentation_template.md).

## Repository structure

```
.
├── README.md                              # this file
├── Documentation_template.md               # full methodology write-up
└── code/
    └── business_entity_resolution/
        ├── src/
        │   ├── normalize.py                # transliteration, abbreviation expansion, skeleton keys
        │   ├── blocking.py                 # adaptive per-country blocking + residual pass
        │   ├── features.py                 # ~24-feature pairwise feature builder
        │   ├── train.py                    # GroupKFold training + threshold search
        │   └── predict.py                  # end-to-end inference
        ├── README.md                        # exact run instructions
        └── requirements.txt                 # pinned dependencies
```

## Tech stack

Python · pandas · NumPy · LightGBM · scikit-learn · RapidFuzz · anyascii

No GPU required for any part of this pipeline.

## Running it yourself

See [`code/business_entity_resolution/README.md`](./code/business_entity_resolution/README.md) for exact setup and run commands (`train.py` then `predict.py`) against the competition's `dataset/train/` and `dataset/test/` files.

## Engineering notes

The real engineering challenge here wasn't the ML — it was making a pipeline built for ~12M+ records survive reliably on free-tier compute (Kaggle). That meant:
- Estimating a join's output size *before* running it, and backing off automatically rather than guessing a fixed memory ceiling.
- Processing each country independently to bound memory, since true matches never cross countries.
- Checkpointing every expensive step immediately, since long-running sessions are never guaranteed to survive.

## License

This project's model (LightGBM) is MIT-licensed and well under the competition's 8-billion-parameter ceiling. No external data, API, or lookup service is used anywhere in this codebase, per the competition's fair-play rules.
