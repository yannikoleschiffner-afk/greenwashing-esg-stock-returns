"""Loughran-McDonald (LM) dictionary scoring - the robustness Talk measure.

Download the LM Master Dictionary CSV from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (free for academic use)
and pass its path with `--lm-dict`.

A sentence is "positive" if, after negation handling, it contains more LM
positive words than LM negative words. Following Loughran & McDonald (2011),
a positive word within three words after a negation (no, not, none, neither,
never, nobody) does not count as positive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

NEGATIONS = {"no", "not", "none", "neither", "never", "nobody"}
NEGATION_WINDOW = 3
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class LMDictionary:
    positive: frozenset[str]
    negative: frozenset[str]

    @classmethod
    def from_csv(cls, path: str | Path) -> "LMDictionary":
        df = pd.read_csv(path, keep_default_na=False, low_memory=False)
        cols = {c.lower(): c for c in df.columns}
        word, pos, neg = cols["word"], cols["positive"], cols["negative"]
        words = df[word].astype(str).str.upper()
        # Non-zero = year the word was added; negative = year it was removed.
        is_pos = pd.to_numeric(df[pos], errors="coerce").fillna(0) > 0
        is_neg = pd.to_numeric(df[neg], errors="coerce").fillna(0) > 0
        return cls(frozenset(words[is_pos]), frozenset(words[is_neg]))


def tone_counts(sentence: str, lm: LMDictionary) -> tuple[int, int]:
    """(positive, negative) LM word counts with negation handling."""
    tokens = [t.lower() for t in _WORD_RE.findall(sentence)]
    pos = neg = 0
    for i, tok in enumerate(tokens):
        up = tok.upper()
        if up in lm.positive:
            if not any(t in NEGATIONS for t in tokens[max(0, i - NEGATION_WINDOW):i]):
                pos += 1
        elif up in lm.negative:
            neg += 1
    return pos, neg


def sentence_tone(sentence: str, lm: LMDictionary) -> str:
    pos, neg = tone_counts(sentence, lm)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"
