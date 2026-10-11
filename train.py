"""
train.py — trains the pair classifier on the training data and picks a
decision threshold.

Pipeline for each country present in dataset/train:
  1. Load and normalize Source-1/2/3 records (see normalize.py).
  2. Block Source-1 against the Source-2/3 pool (see blocking.py), using an
     adaptive MAXB plus a residual pass for weakly-covered entities.
  3. Label each candidate pair against train_ground_truth.tsv.
  4. Build the ~24-column feature matrix (see features.py).

All countries' feature rows are then concatenated and a single LightGBM
classifier is trained across all of them, with:
  - 5-fold GroupKFold (folds split by Source-1 entity, so a single entity's
    candidates never straddle the train/validation boundary within a fold)
    to obtain honest out-of-fold probabilities,
  - a macro-F0.5 threshold search on those out-of-fold probabilities,
    computed EXACTLY as the competition scores it (per Source-1 entity,
    macro-averaged, singletons scoring 1.0 for a correct empty prediction),
  - and a final model refit on 100% of the available training pairs (this is
    the model artifact predict.py actually loads for inference).

Usage:
    python src/train.py --data-dir dataset --out-dir artifacts

Outputs (written to --out-dir):
    model.pkl            — the final, fully-fit LightGBM model
    threshold.json        — {"threshold": <float>, "oof_f05": <float>}
    feature_names.json     — the fixed feature column order (see features.py)
"""

import argparse
import gc
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from normalize import name_norm, addr_norm
from blocking import adaptive_block, residual_pass
from features import build_features, FEATURE_NAMES

# Per-country blocking settings. These were tuned by observing the
# ESTIMATED-pair-count backoff behaviour of adaptive_block on the full
# training data (see Documentation_template.md section 3); they are starting
# targets, not hard limits — adaptive_block will back off automatically if a
# target proves too large for the max_estimated_pairs budget.
TARGET_MAXB = {"US": 300, "India": 250}
DEFAULT_TARGET_MAXB = 250   # used for any country not explicitly listed
TOPK = 30


def _read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _load_and_normalize(data_dir, split):
    s1 = _read_tsv(os.path.join(data_dir, split, f"{split}_source1.tsv"))
    s2 = _read_tsv(os.path.join(data_dir, split, f"{split}_source2.tsv"))
    s3 = _read_tsv(os.path.join(data_dir, split, f"{split}_source3.tsv"))
    for df in (s1, s2, s3):
        df["nn"] = [name_norm(x) for x in df.business_name]
        df["aa"] = [addr_norm(x) for x in df.business_address]
        df["combo"] = df["nn"] + " " + df["aa"]
    return s1, s2, s3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Loading and normalizing training data ===")
    s1, s2, s3 = _load_and_normalize(args.data_dir, "train")
    pool = pd.concat([s2, s3], ignore_index=True)
    ground_truth = _read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    truth = {
        row.source1_entity_id: {x for x in row.matched_entity_ids.split(",") if x}
        for row in ground_truth.itertuples()
    }
    del s2, s3, ground_truth
    gc.collect()

    X_parts, y_parts, group_parts = [], [], []
    country_offset = 0
    for country in sorted(s1.country.unique()):
        s1c = s1[s1.country == country].reset_index(drop=True)
        pc = pool[pool.country == country].reset_index(drop=True)
        target_maxb = TARGET_MAXB.get(country, DEFAULT_TARGET_MAXB)

        pairs = adaptive_block(s1c, pc, target_maxb, TOPK, tag=country)
        pairs = residual_pass(s1c, pc, pairs, min_cands=3, loose_maxb=2000, tag=country)

        s1_ids, pool_ids = s1c.entity_id.values, pc.entity_id.values
        s1_id_set, pool_id_set = set(s1_ids), set(pool_ids)
        true_pairs_local = [
            (a, b) for a, matches in truth.items() if a in s1_id_set
            for b in matches if b in pool_id_set
        ]
        s1_pos = pd.Series(np.arange(len(s1c)), index=s1_ids)
        pool_pos = pd.Series(np.arange(len(pc)), index=pool_ids)
        true_pair_keys = {
            int(s1_pos[a]) * len(pc) + int(pool_pos[b]) for a, b in true_pairs_local
        }
        pair_keys = pairs.i_s1.values.astype(np.int64) * len(pc) + pairs.i_p.values.astype(np.int64)
        pairs["y"] = np.fromiter((k in true_pair_keys for k in pair_keys), bool, len(pairs))
        print(f"{country}: positive rate = {pairs.y.mean():.4f}")

        X, pairs = build_features(pairs, s1c, pc)
        X_parts.append(X)
        y_parts.append(pairs.y.values)
        # Offset each country's i_s1 index range so GroupKFold groups never
        # collide with another country's entities of the same positional index.
        group_parts.append(pairs.i_s1.values + country_offset)
        country_offset += 10_000_000
        del s1c, pc, pairs, X
        gc.collect()

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    del X_parts, y_parts, group_parts
    gc.collect()
    print(f"Combined training data: {X.shape}, positive rate {y.mean():.4f}")

    print("=== Training (5-fold GroupKFold) for out-of-fold threshold search ===")
    oof = np.zeros(len(y), dtype=np.float32)
    for fold_idx, (train_idx, val_idx) in enumerate(GroupKFold(5).split(X, y, groups)):
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.06, num_leaves=63,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            max_bin=127, n_jobs=-1, verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        print(f"fold {fold_idx} done")

    # Threshold search using the EXACT competition metric: macro F0.5,
    # computed per Source-1 entity (a global group index here), singletons
    # scoring 1.0 for a correctly-empty prediction.
    n_entities = int(groups.max()) + 1
    n_true = np.bincount(groups[y], minlength=n_entities)
    best_threshold, best_f05 = 0.5, -1.0
    for threshold in np.arange(0.30, 0.91, 0.05):
        keep = oof >= threshold
        n_pred = np.bincount(groups[keep], minlength=n_entities)
        n_correct = np.bincount(groups[keep][y[keep]], minlength=n_entities).astype(float)
        precision = np.divide(n_correct, n_pred, out=np.zeros(n_entities), where=n_pred > 0)
        recall = np.divide(n_correct, n_true, out=np.zeros(n_entities), where=n_true > 0)
        denom = 0.25 * precision + recall
        f05 = np.divide(1.25 * precision * recall, denom, out=np.zeros(n_entities), where=denom > 0)
        f05[(n_true == 0) & (n_pred == 0)] = 1.0
        macro_f05 = f05.mean()
        print(f"threshold={threshold:.2f}  macro_F0.5={macro_f05:.4f}")
        if macro_f05 > best_f05:
            best_threshold, best_f05 = float(threshold), float(macro_f05)
    print(f"BEST threshold={best_threshold:.2f}  macro_F0.5={best_f05:.4f}")

    print("=== Refitting final model on 100% of training pairs ===")
    final_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.06, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        max_bin=127, n_jobs=-1, verbose=-1,
    )
    final_model.fit(X, y)

    joblib.dump(final_model, os.path.join(args.out_dir, "model.pkl"))
    with open(os.path.join(args.out_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": best_threshold, "oof_f05": best_f05}, f, indent=2)
    with open(os.path.join(args.out_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Saved model, threshold, and feature schema to {args.out_dir}/")


if __name__ == "__main__":
    main()
