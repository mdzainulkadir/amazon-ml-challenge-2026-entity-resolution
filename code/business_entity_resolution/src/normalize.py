"""
normalize.py — text normalization for business_name and business_address fields.

Handles:
  - Transliteration of non-Latin scripts (Devanagari, Gujarati, Telugu, Tamil, etc.)
    and accented Latin (French) to plain ASCII, via `anyascii` (MIT-licensed,
    no external network lookups — pure offline string transformation).
  - Lowercasing and punctuation stripping.
  - Legal-suffix / common-word abbreviation expansion, for English, and French
    (SARL, SAS, etc.), since the test set introduces France as an unseen country.
  - Digit-for-letter "leet" correction inside name tokens only (e.g. "Te1ecom"
    -> "Telecom"), a cheap fix for a common obfuscation-style typo pattern
    observed in the training data.
  - A consonant "skeleton" transform (vowels stripped, repeated letters
    collapsed) used as one of several blocking keys — it is deliberately lossy,
    so that transliteration variants of the same word collide on the same key.

This module has no dependency on any other file in this package and is safe to
import standalone.
"""

import re
from anyascii import anyascii

# Abbreviation expansion table. Extend this per-country as new patterns are
# found in errors analysis; keys must already be lowercase, whitespace-only
# tokens (post punctuation-stripping).
ABBR = {
    # English / US
    "rd": "road", "ave": "avenue", "blvd": "boulevard", "dr": "drive",
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "co": "company",
    # French
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "sasu": "societe par actions simplifiee unipersonnelle",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "ei": "entreprise individuelle",
    "bd": "boulevard", "av": "avenue", "pl": "place", "r": "rue",
    "che": "chemin", "imp": "impasse", "fg": "faubourg",
}

# Tokens that carry little discriminative value for entity matching (generic
# legal/connective words). Used to de-prioritize blocking keys and down-weight
# token-overlap features, NOT to delete the tokens outright.
LEGAL = {
    "incorporated", "corporation", "limited", "private", "company", "llc",
    "llp", "the", "and", "of",
    "societe", "responsabilite", "limitee", "par", "actions", "simplifiee",
    "unipersonnelle", "entreprise", "individuelle",
}

_VOWEL_RE = re.compile(r"[aeiouy]")
_LEET_TABLE = str.maketrans("015348", "olseab")
_PUNCT_RE = re.compile(r"[^\w\s]")


def norm(text: str) -> str:
    """Generic normalization for a raw string field (name or address).

    Steps: transliterate to ASCII -> lowercase -> replace '&' with 'and' ->
    strip punctuation -> expand known abbreviations token-by-token.
    """
    s = anyascii(str(text)).lower().replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    return " ".join(ABBR.get(tok, tok) for tok in s.split())


def _fix_leet_token(tok: str) -> str:
    """Undo digit-for-letter substitution, but ONLY on tokens that mix letters
    and digits (e.g. 'te1ecom') — a token that is purely numeric (a house
    number, a PIN code) must never be touched here.
    """
    if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
        return tok.translate(_LEET_TABLE)
    return tok


def name_norm(business_name: str) -> str:
    """Normalization specialized for business_name: applies `norm`, then the
    leet-token fix. Use this (not the bare `norm`) for the name field.
    """
    return " ".join(_fix_leet_token(t) for t in norm(business_name).split())


def addr_norm(business_address: str) -> str:
    """Normalization for business_address. Currently identical to `norm`
    (kept as a separate function so address-specific rules can be added later
    without touching name handling), plus mapping the literal string "null"
    (produced when the source address is missing) to an empty string so
    downstream code can test for a missing address with a single `== ""`.
    """
    out = norm(business_address)
    return "" if out == "null" else out


def skel(token: str) -> str:
    """Consonant-skeleton transform: strips vowels and collapses immediately-
    repeated characters. Used only as one specific blocking key, since it is
    intentionally lossy (many different real words can collapse to the same
    skeleton) — it exists to catch transliteration variants such as
    "Shree"/"Shri" -> "shr", not to be a general-purpose feature.
    """
    stripped = _VOWEL_RE.sub("", token)
    collapsed = re.sub(r"(.)\1+", r"\1", stripped)
    return collapsed or token
