"""
features.py — builds the pairwise feature matrix consumed by the classifier
in train.py / predict.py.

Feature families (see Documentation_template.md section 4):
  - Name string-similarity: Levenshtein ratio, token-sort ratio, token-set
    ratio, partial ratio, Jaro-Winkler similarity, normalized Levenshtein.
  - Address string-similarity: the same ratio family applied to addresses.
  - IDF-weighted token Jaccard (name) and numeric-token Jaccard (address) —
    down-weights common words / gives credit for shared distinctive tokens.
  - Missing-address flags, name length difference.
  - Context features: for each candidate, how it ranks / how large its score
    gap is versus (a) other candidates for the same Source-1 entity and
    (b) other Source-1 entities competing for the same pool record. These
    features let the model learn one-to-one assignment behaviour directly,
    rather than relying solely on the post-hoc greedy assignment rule.

`build_features` takes the candidate-pairs dataframe produced by
`blocking.adaptive_block` (+ `residual_pass`) together with the two source
dataframes, and returns (X, pairs) where X is a numpy float32 array with a
FIXED column order — this order must be identical between train.py and
predict.py, since the trained model's feature indices are order-dependent.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import LEGAL

# Fixed feature column order — DO NOT reorder without retraining the model.
FEATURE_NAMES = [
    "n_ratio", "n_sort", "n_part", "n_jw", "n_lev",
    "a_ratio", "a_sort", "a_part", "a_lev",
    "idf_j", "num_j", "n_lendiff", "a_miss_pool", "a_miss_s1",
    "nk", "ns", "ads", "score",
    "gap_s1", "rank_s1", "gap_p", "rank_p", "n_p", "n_s1",
]


def _idf_jaccard(token_set_a, token_set_b, weights, default_weight):
    if not token_set_a or not token_set_b:
        return 0.0
    union_weight = sum(weights.get(t, default_weight) for t in token_set_a | token_set_b)
    if union_weight == 0:
        return 0.0
    inter_weight = sum(weights.get(t, default_weight) for t in token_set_a & token_set_b)
    return inter_weight / union_weight


def _numeric_jaccard(digit_set_a, digit_set_b):
    if not digit_set_a or not digit_set_b:
        return -1.0  # sentinel: "no numeric tokens to compare", distinct from 0.0 ("compared, no overlap")
    return len(digit_set_a & digit_set_b) / len(digit_set_a | digit_set_b)


def build_features(pairs: pd.DataFrame, s1c: pd.DataFrame, pc: pd.DataFrame):
    """Compute the fixed-order feature matrix for every row of `pairs`.

    `pairs` must have integer columns i_s1 (positional index into s1c) and
    i_p (positional index into pc), plus nk/ns/ads/score already populated by
    the blocking stage (adaptive_block always fills these).

    Returns (X, pairs) where X is shape (len(pairs), len(FEATURE_NAMES)) and
    `pairs` is the input dataframe with the added context-feature columns
    (gap_s1, rank_s1, gap_p, rank_p, n_p, n_s1), in case the caller wants them
    for downstream analysis or writing intermediate diagnostics.
    """
    n1v, a1v = s1c.nn.values, s1c.aa.values
    n2v, a2v = pc.nn.values, pc.aa.values

    total_n = len(s1c) + len(pc)
    doc_freq = Counter()
    for v in pd.concat([s1c.nn, pc.nn]):
        doc_freq.update({t for t in v.split() if t not in LEGAL})
    weights = {t: math.log(total_n / (1 + c)) for t, c in doc_freq.items()}
    default_weight = math.log(total_n)

    name_tokens_1 = [frozenset(t for t in s.split() if t not in LEGAL) for s in n1v]
    name_tokens_2 = [frozenset(t for t in s.split() if t not in LEGAL) for s in n2v]
    digit_tokens_1 = [frozenset(t for t in s.split() if any(ch.isdigit() for ch in t)) for s in a1v]
    digit_tokens_2 = [frozenset(t for t in s.split() if any(ch.isdigit() for ch in t)) for s in a2v]

    # Context features: computed once via groupby over the WHOLE pairs frame.
    by_s1 = pairs.groupby("i_s1").score
    by_pool = pairs.groupby("i_p").score
    pairs = pairs.copy()
    pairs["gap_s1"] = by_s1.transform("max") - pairs.score
    pairs["rank_s1"] = by_s1.rank(ascending=False, method="min")
    pairs["gap_p"] = by_pool.transform("max") - pairs.score
    pairs["rank_p"] = by_pool.rank(ascending=False, method="min")
    pairs["n_p"] = by_pool.transform("size")
    pairs["n_s1"] = by_s1.transform("size")

    ii, jj = pairs.i_s1.values, pairs.i_p.values
    cols = {}
    for name, fn in (
        ("n_ratio", fuzz.ratio), ("n_sort", fuzz.token_sort_ratio),
        ("n_part", fuzz.partial_ratio), ("n_jw", JaroWinkler.similarity),
        ("n_lev", Levenshtein.normalized_similarity),
    ):
        cols[name] = np.fromiter((fn(n1v[x], n2v[y]) for x, y in zip(ii, jj)), np.float32, len(ii))
    for name, fn in (
        ("a_ratio", fuzz.ratio), ("a_sort", fuzz.token_sort_ratio),
        ("a_part", fuzz.partial_ratio), ("a_lev", Levenshtein.normalized_similarity),
    ):
        cols[name] = np.fromiter((fn(a1v[x], a2v[y]) for x, y in zip(ii, jj)), np.float32, len(ii))

    cols["idf_j"] = np.fromiter(
        (_idf_jaccard(name_tokens_1[x], name_tokens_2[y], weights, default_weight) for x, y in zip(ii, jj)),
        np.float32, len(ii),
    )
    cols["num_j"] = np.fromiter(
        (_numeric_jaccard(digit_tokens_1[x], digit_tokens_2[y]) for x, y in zip(ii, jj)),
        np.float32, len(ii),
    )
    cols["n_lendiff"] = np.abs(np.array([len(n1v[x]) - len(n2v[y]) for x, y in zip(ii, jj)], np.float32))
    cols["a_miss_pool"] = np.array([a2v[y] == "" for y in jj], np.float32)
    cols["a_miss_s1"] = np.array([a1v[x] == "" for x in ii], np.float32)

    for name in ("nk", "ns", "ads", "score", "gap_s1", "rank_s1", "gap_p", "rank_p", "n_p", "n_s1"):
        cols[name] = pairs[name].values.astype(np.float32)

    # Enforce the fixed column order defined in FEATURE_NAMES.
    X = np.column_stack([cols[name] for name in FEATURE_NAMES])
    return X, pairs
