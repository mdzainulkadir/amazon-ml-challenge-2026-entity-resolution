"""
blocking.py — candidate generation (blocking) for one country at a time.

Design summary (see Documentation_template.md section 3 for the full writeup):

  1. Build several cheap "blocking keys" per record from its normalized name
     and address (rarest-token keys, prefix keys, consonant-skeleton keys,
     address-number keys, rare-address-word keys).
  2. Join Source-1 records to the Source-2/3 pool on shared keys.
  3. Any key shared by more than `MAXB` pool records is dropped before the
     join — an extremely common key (e.g. a generic word) would otherwise
     create a huge, low-precision block.
  4. `MAXB` is not a fixed constant: `adaptive_block` estimates the resulting
     pair count for a target MAXB from key-frequency statistics BEFORE doing
     the expensive join, and backs off geometrically if the estimate exceeds
     a compute/memory budget. This keeps blocking safe to run at full scale
     without manual per-country tuning.
  5. Candidates are ranked by a combined name+address similarity score and
     trimmed to the top `TOPK` per Source-1 entity — this final, ranked list
     is exactly what should be written to candidate_pairs.tsv.
  6. `residual_pass` gives a second, more permissive blocking attempt ONLY to
     Source-1 entities that still have very few candidates after step 5 —
     recovering recall for hard cases without inflating the candidate set
     for the (large) majority of already-well-served entities. This keeps
     the overall candidate set efficient, which the competition organizers
     have indicated is itself part of the final ranking criteria.

This module is CPU-only (no GPU dependency) and uses only pandas / numpy /
rapidfuzz, all permissively licensed and far under the 8B-parameter model
constraint (blocking itself uses no learned model at all).
"""

import gc
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import LEGAL


def _name_tokens(nn: str):
    return [t for t in nn.split() if t not in LEGAL and len(t) >= 2]


def dfreq_for(name_series):
    """Corpus-wide frequency count of name tokens, for ranking 'rarest first'."""
    c = Counter()
    for v in name_series:
        c.update(set(_name_tokens(v)))
    return c


def adfreq_for(addr_series):
    """Corpus-wide frequency count of alphabetic address words (len>=3)."""
    c = Counter()
    for v in addr_series:
        c.update({t for t in v.split() if t.isalpha() and len(t) >= 3})
    return c


def _build_keys(nn: str, aa: str, dfreq: dict, adfreq: dict):
    """Construct the list of blocking keys for a single (normalized name,
    normalized address) pair. See module docstring for a description of each
    key family.
    """
    from normalize import skel

    tokens = sorted(_name_tokens(nn), key=lambda t: (dfreq.get(t, 0), t))
    top = ([t for t in tokens if dfreq.get(t, 0) >= 2] or tokens)[:3]

    addr_tokens = aa.split()
    numeric = [t for t in addr_tokens if any(ch.isdigit() for ch in t)][:4]
    rare_words = sorted(
        {t for t in addr_tokens if t.isalpha() and len(t) >= 3},
        key=lambda t: (adfreq.get(t, 0), t),
    )
    rare_words = [t for t in rare_words if adfreq.get(t, 0) >= 2][:2]

    prefix3 = top[0][:3] if top else ""
    keys = []
    for t in top:
        keys += [f"n1|{t}", f"n2|{t[:4]}", f"n3|{skel(t)}"]
    if len(top) >= 2:
        keys.append(f"n4|{top[0][:3]}|{top[1][:3]}")
    keys += [f"a1|{prefix3}|{n}" for n in numeric]
    keys += [f"a2|{n}" for n in numeric if len(n) >= 5]
    if len(rare_words) == 2:
        keys.append(f"a3|{rare_words[0]}|{rare_words[1]}")
    keys += [f"a4|{n}|{skel(w)}" for n in numeric for w in rare_words]
    return keys


def key_table(df: pd.DataFrame, dfreq: dict, adfreq: dict, chunk: int = 200_000) -> pd.DataFrame:
    """Build a long-format (row_index, hashed_key) table for a dataframe with
    'nn' (normalized name) and 'aa' (normalized address) columns. Processed in
    chunks so peak memory stays bounded regardless of the input's total size.
    """
    parts = []
    for start in range(0, len(df), chunk):
        d = df.iloc[start:start + chunk]
        keys = [_build_keys(n, a, dfreq, adfreq) for n, a in zip(d.nn, d.aa)]
        t = pd.DataFrame(
            {"i": np.arange(start, start + len(d), dtype=np.int32), "key": keys}
        ).explode("key").dropna()
        t["h"] = pd.util.hash_array(t["key"].to_numpy())
        parts.append(t[["i", "h"]])
    return pd.concat(parts, ignore_index=True).drop_duplicates()


# ---------------------------------------------------------------------------
# Parallel rapidfuzz scoring (used to rank candidates before the TOPK trim)
# ---------------------------------------------------------------------------
_WORKER_STATE = {}


def _init_worker(n1v, a1v, n2v, a2v):
    _WORKER_STATE["n1"], _WORKER_STATE["a1"] = n1v, a1v
    _WORKER_STATE["n2"], _WORKER_STATE["a2"] = n2v, a2v


def _score_chunk(args):
    ii, jj = args
    n1, a1 = _WORKER_STATE["n1"], _WORKER_STATE["a1"]
    n2, a2 = _WORKER_STATE["n2"], _WORKER_STATE["a2"]
    name_sim = np.fromiter(
        (fuzz.token_set_ratio(n1[x], n2[y]) for x, y in zip(ii, jj)), np.float32, len(ii)
    )
    addr_sim = np.fromiter(
        (fuzz.token_set_ratio(a1[x], a2[y]) for x, y in zip(ii, jj)), np.float32, len(ii)
    )
    return name_sim, addr_sim


def parallel_score(n1v, a1v, n2v, a2v, ii, jj, nproc: int = 4):
    """Compute rapidfuzz token_set_ratio for name and address, split across
    `nproc` worker processes for speed on multi-core machines.
    """
    splits = np.array_split(np.arange(len(ii)), nproc)
    chunks = [(ii[s], jj[s]) for s in splits]
    with mp.Pool(nproc, initializer=_init_worker, initargs=(n1v, a1v, n2v, a2v)) as pool:
        results = pool.map(_score_chunk, chunks)
    return (
        np.concatenate([r[0] for r in results]),
        np.concatenate([r[1] for r in results]),
    )


def adaptive_block(s1c: pd.DataFrame, pc: pd.DataFrame, target_maxb: int, topk: int,
                    tag: str = "", max_estimated_pairs: int = 120_000_000) -> pd.DataFrame:
    """Block one country's Source-1 slice (`s1c`) against its Source-2/3 pool
    slice (`pc`). Both dataframes must already have 'nn'/'aa'/'entity_id'
    columns (see normalize.py).

    Automatically searches downward from `target_maxb` for the largest key-
    frequency cap whose ESTIMATED resulting pair count stays under
    `max_estimated_pairs`, before doing the (expensive) actual join. This is
    what makes the function safe to run unattended on country pools of very
    different sizes without manual tuning per country.

    Returns a dataframe with columns: i_s1, i_p, nk, ns, ads, score
    (indices are positional into s1c / pc respectively).
    """
    print(f"{tag}: s1={len(s1c)} pool={len(pc)}")
    empty = pd.DataFrame(columns=["i_s1", "i_p", "nk", "ns", "ads", "score"])
    if len(s1c) == 0 or len(pc) == 0:
        return empty

    dfreq = dfreq_for(pd.concat([s1c.nn, pc.nn]))
    adfreq = adfreq_for(pd.concat([s1c.aa, pc.aa]))
    t1 = key_table(s1c, dfreq, adfreq)
    tp_full = key_table(pc, dfreq, adfreq)
    key_sizes = tp_full.groupby("h").size()

    maxb = target_maxb
    tp = None
    while maxb >= 25:
        tp = tp_full[tp_full.h.isin(key_sizes.index[key_sizes <= maxb])]
        t1_counts = t1.h.value_counts()
        tp_counts = tp.h.value_counts()
        common = t1_counts.index.intersection(tp_counts.index)
        estimate = int((t1_counts.reindex(common) * tp_counts.reindex(common)).sum()) if len(common) else 0
        print(f"{tag}: MAXB={maxb} -> estimated pairs ~{estimate:,}")
        if estimate <= max_estimated_pairs:
            break
        maxb = int(maxb * 0.6)
    del tp_full
    gc.collect()

    joined_chunks = []
    for start in range(0, len(s1c), 40_000):
        merged = t1[(t1.i >= start) & (t1.i < start + 40_000)].merge(tp, on="h", suffixes=("_s1", "_p"))
        joined_chunks.append(
            merged.groupby(["i_s1", "i_p"], sort=False).size().rename("nk").reset_index()
        )
        del merged
    pairs = pd.concat(joined_chunks, ignore_index=True) if joined_chunks else empty.copy()
    del joined_chunks, t1, tp
    gc.collect()
    print(f"{tag}: ACTUAL PAIR COUNT = {len(pairs):,} (used MAXB={maxb})")
    if len(pairs) == 0:
        return pairs

    n1v, a1v = s1c.nn.values, s1c.aa.values
    n2v, a2v = pc.nn.values, pc.aa.values
    ii, jj = pairs.i_s1.values, pairs.i_p.values
    name_sim, addr_sim = parallel_score(n1v, a1v, n2v, a2v, ii, jj, nproc=4)
    pairs["ns"], pairs["ads"] = name_sim, addr_sim
    pairs["score"] = pairs.ns + pairs.ads + 2 * pairs.nk
    pairs = (
        pairs.sort_values(["i_s1", "score"], ascending=[True, False])
        .groupby("i_s1", sort=False)
        .head(topk)
    )
    print(f"{tag}: kept top-{topk} -> {len(pairs):,} pairs")
    return pairs.reset_index(drop=True)


def residual_pass(s1c: pd.DataFrame, pc: pd.DataFrame, pairs: pd.DataFrame,
                   min_cands: int = 3, loose_maxb: int = 2000, tag: str = "") -> pd.DataFrame:
    """Give a second, looser blocking attempt ONLY to Source-1 entities whose
    candidate count from `adaptive_block` is below `min_cands`. Because this
    subset is small (typically well under 2% of a country's entities), it can
    afford a much larger MAXB without materially increasing the total
    candidate-set size — the goal is recall recovery for hard cases, not a
    blanket loosening of blocking for everyone.
    """
    counts = pairs.groupby("i_s1").size() if len(pairs) else pd.Series(dtype=int)
    weak_idx = [i for i in range(len(s1c)) if counts.get(i, 0) < min_cands]
    print(f"{tag} residual: {len(weak_idx)} weak entities out of {len(s1c)}")
    if not weak_idx:
        return pairs

    s1_weak = s1c.iloc[weak_idx].reset_index(drop=True)
    extra = adaptive_block(
        s1_weak, pc, loose_maxb, topk=30, tag=f"{tag}-residual", max_estimated_pairs=40_000_000
    )
    if len(extra) == 0:
        return pairs

    index_map = {new: old for new, old in enumerate(weak_idx)}
    extra["i_s1"] = extra["i_s1"].map(index_map)
    return pd.concat([pairs, extra], ignore_index=True).drop_duplicates(["i_s1", "i_p"])
