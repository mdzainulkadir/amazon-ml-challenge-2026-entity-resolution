"""
predict.py — end-to-end inference: test data -> matching_results.tsv +
candidate_pairs.tsv.

Runs, per country present in the test set (including any country never seen
in training — `country` is treated throughout as an open string label, never
a hardcoded set, so e.g. France is handled with no special-cased code path):

  1. Load and normalize test Source-1/2/3 records.
  2. Block (adaptive_block + residual_pass), exactly as in training.
  3. Build the same fixed-order feature matrix as training.
  4. Score every candidate pair with the trained model.
  5. Apply the decision rule: keep pairs at or above the trained threshold,
     then enforce one-to-one assignment (each Source-2/3 record is kept only
     by its single highest-probability Source-1 entity) — justified by the
     training-data finding that no pool record is a true match for more than
     one Source-1 entity.
  6. Write BOTH required output files. candidate_pairs.tsv contains exactly
     the candidate set the model scored in step 4 (the top-K, post-blocking
     list) — matching_results.tsv is guaranteed to be a subset of it, since
     matches are only ever selected from that same scored set.

Usage:
    python src/predict.py --data-dir dataset --artifacts-dir artifacts --out-dir output

Outputs (written to --out-dir):
    matching_results.tsv
    candidate_pairs.tsv
"""

import argparse
import gc
import json
import os

import joblib
import numpy as np
import pandas as pd

from normalize import name_norm, addr_norm
from blocking import adaptive_block, residual_pass
from features import build_features

TARGET_MAXB = {"US": 300, "India": 250}
DEFAULT_TARGET_MAXB = 250
TOPK = 30
MIN_RESIDUAL_CANDS = 3
RESIDUAL_LOOSE_MAXB = 2000


def _read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _load_and_normalize_test(data_dir):
    s1 = _read_tsv(os.path.join(data_dir, "test", "test_source1.tsv"))
    s2 = _read_tsv(os.path.join(data_dir, "test", "test_source2.tsv"))
    s3 = _read_tsv(os.path.join(data_dir, "test", "test_source3.tsv"))
    for df in (s1, s2, s3):
        df["nn"] = [name_norm(x) for x in df.business_name]
        df["aa"] = [addr_norm(x) for x in df.business_address]
    return s1, s2, s3


def _write_id_list_tsv(entity_to_ids: dict, path: str, list_column_name: str):
    out = pd.DataFrame({
        "source1_entity_id": list(entity_to_ids.keys()),
        list_column_name: [",".join(sorted(ids)) for ids in entity_to_ids.values()],
    })
    out.to_csv(path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = joblib.load(os.path.join(args.artifacts_dir, "model.pkl"))
    with open(os.path.join(args.artifacts_dir, "threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    print(f"Loaded model. Using decision threshold = {threshold}")

    print("=== Loading and normalizing test data ===")
    test_s1, test_s2, test_s3 = _load_and_normalize_test(args.data_dir)
    test_pool = pd.concat([test_s2, test_s3], ignore_index=True)
    del test_s2, test_s3
    gc.collect()
    print("Countries in test Source-1:", test_s1.country.value_counts().to_dict())

    all_matches: dict = {}
    all_candidates: dict = {}

    for country in test_s1.country.unique():
        target_maxb = TARGET_MAXB.get(country, DEFAULT_TARGET_MAXB)
        s1c = test_s1[test_s1.country == country].reset_index(drop=True)
        pc = test_pool[test_pool.country == country].reset_index(drop=True)

        pairs = adaptive_block(s1c, pc, target_maxb, TOPK, tag=country)
        pairs = residual_pass(
            s1c, pc, pairs, min_cands=MIN_RESIDUAL_CANDS,
            loose_maxb=RESIDUAL_LOOSE_MAXB, tag=country,
        )

        # Every Source-1 entity must appear in the output, even with zero
        # candidates (an empty list = a singleton prediction).
        for entity_id in s1c.entity_id.values:
            all_matches[entity_id] = set()
            all_candidates[entity_id] = set()
        if len(pairs) == 0:
            continue

        X, pairs = build_features(pairs, s1c, pc)
        pairs["p"] = model.predict_proba(X)[:, 1]

        s1_ids, pool_ids = s1c.entity_id.values, pc.entity_id.values
        for row in pairs.itertuples():
            all_candidates[s1_ids[row.i_s1]].add(pool_ids[row.i_p])

        kept = pairs[pairs.p >= threshold]
        # One-to-one assignment: each pool record goes to its single
        # highest-probability Source-1 entity.
        kept = kept.sort_values("p", ascending=False).drop_duplicates("i_p")
        for row in kept.itertuples():
            all_matches[s1_ids[row.i_s1]].add(pool_ids[row.i_p])

        print(f"{country}: {len(kept)} final matches assigned")
        del s1c, pc, pairs, X
        gc.collect()

    print(f"Done. Total Source-1 entities covered: {len(all_matches)}")
    _write_id_list_tsv(all_matches, os.path.join(args.out_dir, "matching_results.tsv"), "matched_entity_ids")
    _write_id_list_tsv(all_candidates, os.path.join(args.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids")

    assert len(all_matches) == len(test_s1), "Every test Source-1 entity must appear exactly once."
    assert all(len(v) == len(set(v)) for v in all_matches.values()), "No duplicate IDs allowed within a list."
    print("Wrote output/matching_results.tsv and output/candidate_pairs.tsv")
    print("Run utils/validate_submission.py against these files before uploading.")


if __name__ == "__main__":
    main()
