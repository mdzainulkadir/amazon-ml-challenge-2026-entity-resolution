# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Wolf Crew
**Team Members:** Md Zainul Kadir, Sameer Ali
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a two-stage entity resolution pipeline: multi-key candidate generation (blocking) per country, followed by a LightGBM binary classifier that scores each candidate pair and applies a probability threshold with one-to-one assignment. Our core innovation is an adaptive, memory-aware blocking strategy that automatically tunes the per-country candidate cap to the largest value that stays within compute limits, combined with a targeted second-pass ("residual") blocking step that gives extra retrieval effort only to entities the first pass under-served — keeping the overall candidate set efficient rather than uniformly large.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis of the training data (2,206,821 Source 1 records; Source 2/3 pools of ~5.0M and ~5.3M records) showed:
- Only 5.58% of Source 1 entities are singletons (no match) — most entities have 2-5 true matches (mean 3.46, max 11).
- Every Source 2/3 record is claimed by at most one Source 1 entity (verified directly on the ground truth) — this lets us apply a one-to-one assignment rule at inference time as a precision safeguard.
- 100% of true matched pairs share the same `country` label in training, so blocking can safely be scoped per-country without losing recall.
- Name noise includes legal-suffix variation (Corp/Corporation, Pvt/Private), transliteration into Devanagari/Gujarati/Telugu/Tamil scripts for Indian entities, digit-for-letter substitution (e.g. "Té1ecom"), and character-level typos.
- Address noise includes missing components, landmark references, reordered tokens, and country-specific abbreviations (US: Rd/Ave/Blvd; French: Bd/Av/R for rue).
- The test set introduces France, a country absent from training entirely — our pipeline treats `country` as an open string label throughout (grouping/scoping logic, never a hardcoded set) so it generalizes to unseen labels without code changes.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (candidate generation → supervised binary matcher → threshold/assignment).

**Core Innovation:** An *adaptive* blocking cap (`MAXB`) chosen automatically per country by estimating the resulting candidate-pair volume from key-frequency statistics before committing to the full join, backing off geometrically until the estimate fits a fixed compute budget. This replaces manual per-country tuning with a repeatable, data-driven procedure, and is combined with a **residual pass**: entities that still have very few candidates after the main pass (e.g. <3) receive a second, more permissive blocking attempt scoped only to that small subset — recovering recall for hard cases without inflating the candidate set for the (large) majority of well-served entities.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** (built per-country, after text normalization):
- Text normalization: `anyascii`-based transliteration to Latin script (handles Devanagari/Gujarati/Telugu/Tamil and French accents), lowercasing, punctuation stripping, legal-suffix/abbreviation expansion (English and French: Corp/Corporation, SARL/société à responsabilité limitée, Rd/Road, Bd/Boulevard, etc.), and digit-for-letter correction in name tokens (e.g. "1" → "l") to counter obfuscation-style typos.
- **Name keys:** rarest name token, rarest-token 4-character prefix, consonant-skeleton of the rarest token (vowel-stripped, survives many transliteration/typo variants), and a 2-token combination key.
- **Address keys:** house-number/PIN-style numeric tokens combined with the rarest name-token prefix, standalone long numeric tokens (postal-code-like), pairs of rare address words, and number+word-skeleton combinations.
- Each key's global frequency is capped (`MAXB`) so overly common keys (which would create huge, low-precision blocks) are excluded from the join.
- **Candidate pairs generated (test set, before scoring):** approximately 43.1M (US), 22.0M (France), 71.1M (India) key-based pairs, further narrowed to the top-30 candidates per Source 1 entity by a combined name+address similarity score before being passed to the classifier — this final, ranked top-30 list is exactly what is written to `candidate_pairs.tsv` and what the model scores at inference.
- **Residual pass:** entities with fewer than 3 candidates after the main pass (typically 0.3%-1.5% of entities per country) are re-blocked with a much looser cap, scoped only to that subset, so the cost of recovering their recall does not affect the candidate set size for the rest of the entities.
- **Ensuring true matches were not lost:** blocking recall was measured directly against training ground truth per country before committing to the full run (US ≈ 94.4%, India ≈ 88.1% at our final settings) — we treat this recall as the honest ceiling on achievable F0.5 and iterated on the key design (adding transliteration/skeleton keys, address-number keys) specifically to close gaps identified by inspecting missed true pairs.

---

## 4. Matching Model

**Features used** (computed per candidate pair, ~24 total):

- **Name features:** Levenshtein ratio, token-sort ratio, token-set ratio, partial ratio, Jaro-Winkler similarity, normalized Levenshtein similarity, length difference, IDF-weighted token Jaccard (down-weights common words, up-weights rare/distinctive tokens shared between the pair).
- **Address features:** the same ratio/similarity family applied to normalized addresses, plus numeric-token Jaccard (captures house-number/PIN-code overlap independent of surrounding text) and a missing-address flag for each side.
- **Context features:** for each candidate, its similarity-score gap and rank relative to the *other* candidates competing for the same Source 1 entity, and relative to the *other* Source 1 entities competing for the same Source 2/3 record — these let the model learn one-to-one assignment behavior directly rather than relying solely on a post-hoc rule.

**Model type:** LightGBM gradient-boosted decision tree classifier, trained with 3-to-5-fold `GroupKFold` cross-validation (folds split by Source 1 entity, preventing any leakage across a single entity's candidates), producing out-of-fold probabilities used for threshold selection, then a final model refit on all available training pairs. LightGBM is MIT-licensed and, at a few hundred trees, is far under the challenge's 8-billion-parameter ceiling.

**Threshold selection method:** we searched thresholds from 0.30 to 0.90 in increments of 0.05 and selected the value maximizing the *macro* F0.5 (computed per Source 1 entity, exactly matching the competition's scoring formula) on out-of-fold predictions, evaluated separately per country to confirm a single global threshold (0.70) performed close to each country's individual optimum. At inference, we additionally apply a one-to-one assignment rule — each Source 2/3 record is kept only by its highest-probability Source 1 entity — motivated by the training-data finding that no record is a true match for more than one Source 1 entity.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro, out-of-fold on training data):** ≈0.967 overall (US ≈0.969-0.970, India ≈0.963-0.964).
- **Public leaderboard score:** 0.9213 (submission after adaptive-blocking and French-specific normalization improvements; an earlier, less-tuned version scored 0.9139).
- **Common false positives (wrong merges):** distinct businesses sharing an address (e.g. different companies registered at the same building) or generic/short business names with high token overlap but different qualifying details — the model's low-frequency-token-weighted features and one-to-one assignment target this failure mode directly.
- **Common false negatives (missed matches):** heavily corrupted or minimal source records (e.g. names reduced to unrelated-looking strings, or missing address fields entirely) where neither name nor address carries enough recoverable signal; and, at the blocking stage, records whose only shared signal is common tokens that exceed our frequency cap and are therefore excluded from candidate generation.

---

## 6. Conclusion

We prioritized a robust, memory-safe, and country-generalizing pipeline (handling France, an unseen country, without any special-cased logic) over a more complex neural architecture, given time constraints. The adaptive blocking cap and residual pass gave a measurable, verifiable recall improvement (India: 86.0% → 88.1%) without manual per-country guesswork, and directly addresses the competition's later-clarified emphasis on efficient, non-bloated candidate generation. Key lessons: candidate generation is the binding constraint on achievable score, per-country processing is essential given that address/name noise patterns differ by locale, and a precision-heavy metric (F0.5) rewards a conservative one-to-one assignment rule as much as raw model accuracy.

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/src/`:
- `normalize.py` — text normalization (transliteration, abbreviation expansion, skeleton keys)
- `blocking.py` — adaptive per-country blocking with automatic MAXB estimation, plus the residual pass for low-coverage entities
- `features.py` — the ~24-feature pair-feature builder
- `train.py` — GroupKFold training, out-of-fold threshold search, final model fit
- `predict.py` — end-to-end inference: blocking → features → scoring → threshold/assignment → `matching_results.tsv` / `candidate_pairs.tsv`

Entry point: `python src/predict.py` (after `python src/train.py` has produced a model artifact), as detailed in `README.md`.

### B. Additional Results

Blocking recall by country (training data, final settings): US ≈94.4%, India ≈88.1%. Positive-pair rate in the final candidate set: ≈12-17% depending on country. Threshold sweep (macro F0.5, out-of-fold): rose from 0.947 at threshold 0.30 to a peak of ≈0.967-0.969 at threshold 0.65-0.70, then declined at higher thresholds as recall losses outweighed precision gains — consistent with the metric's precision weighting still requiring adequate recall to maximize the harmonic mean.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
