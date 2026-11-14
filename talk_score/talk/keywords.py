"""Environmental keyword list.

Used twice:
  * Loughran-McDonald method: a sentence is "environmental" if it matches.
  * FinBERT method: only matching sentences are sent to FinBERT-ESG (a
    prefilter to keep GPU time manageable). FinBERT then makes the actual
    environmental / not-environmental decision. `score --no-prefilter` on a
    sample checks how many FinBERT-environmental sentences the prefilter misses.

The list errs on the side of recall. Business uses of ambiguous words
("business environment", "sustainable growth", "carbon steel") are excluded.
"""
from __future__ import annotations

import re

_TERMS = [
    r"environmental(?:ly)?",
    r"environment",  # filtered by _business_environment below
    r"climate",  # filtered by _NON_ENV_PHRASES below
    r"global warming",
    r"carbon(?! black| steel| fiber| fibre)",
    r"decarboni[sz]\w*",
    r"low[- ]carbon",
    r"net[- ]zero",
    r"emissions?",
    r"greenhouse",
    r"ghg",
    r"co2e?",
    r"scope [123]",
    r"renewables?",
    r"solar",
    r"wind (?:energy|power|farms?|turbines?|generation|projects?)",
    r"clean (?:energy|power|fuels?|technolog\w*|tech|air|water)",
    r"green (?:bonds?|buildings?|energy|power|products?|technolog\w*|hydrogen|financing)",
    r"energy[- ]efficien\w*",
    r"fuel[- ]efficien\w*",
    r"electric vehicles?",
    r"fossil fuels?",
    r"sustainab\w*",  # filtered by _financial_sustainability below
    r"pollut\w*",
    r"wastes?",
    r"wastewater",
    r"recycl\w*",
    r"circular economy",
    r"water (?:use|usage|consumption|conservation|stewardship|quality|scarcity|withdrawals?|efficiency|management)",
    r"biodiversity",
    r"deforest\w*",
    r"habitats?",
    r"endangered species",
    r"ecolog\w*",
    r"conservation",  # filtered by _NON_ENV_PHRASES below
    r"natural resources?",
    r"hazardous (?:materials?|substances?|wastes?|chemicals?)",
    r"toxic",
    r"(?:oil|chemical) spills?",
    r"epa",
    r"paris (?:agreement|accord|climate)",
    r"tcfd",
    r"cap[- ]and[- ]trade",
    r"leed",
]

KEYWORD_RE = re.compile(r"\b(?:" + "|".join(_TERMS) + r")\b", re.I)

_BUSINESS_MODIFIERS = {
    "business", "economic", "macroeconomic", "competitive", "regulatory", "operating",
    "rate", "pricing", "market", "political", "legal", "tax", "control", "it",
    "technology", "computing", "cloud", "work", "working", "retail", "credit", "lending",
    "trading", "financial", "inflationary", "geopolitical", "current", "challenging",
    "difficult", "uncertain", "volatile", "security", "threat", "customer", "consumer",
    "labor", "test", "testing", "production", "development", "virtual", "digital",
    "banking", "reimbursement", "healthcare", "funding", "interest", "investment",
    "industry", "enforcement", "litigation", "sales", "selling", "demand", "global",
    "low", "high", "rising", "changing", "evolving", "compliance", "hybrid", "remote",
    "office", "software", "network", "computer", "data", "processing", "hosting",
    "information", "systems", "system", "application", "user", "insurance", "capital",
    "networked", "respectful", "inclusive", "collaborative", "supportive", "diverse",
    "learning", "team", "cybersecurity", "cyber", "threat", "workplace", "safe", "secure",
    "hosted", "multi", "cloud-based", "on-premise", "premises", "lending", "trade",
    "controls", "this", "such",
}

# Phrases where an environmental keyword has a business meaning.
_NON_ENV_PHRASES = re.compile(
    r"\b(?:business|economic|political|investment|regulatory|social|credit|financial|"
    r"market|geopolitical|current|operating)\s+climate\b"
    r"|\bcapital\s+conservation\b",
    re.I,
)

_FIN_SUSTAINABILITY = re.compile(
    r"\bsustainab\w*\s+(?:growth|competitive|revenues?|earnings|returns?|profit\w*|cash|"
    r"dividends?|margins?|performance|basis|levels?|rates?|value|shareholder)\b",
    re.I,
)


def _is_business_environment(text: str, m: re.Match) -> bool:
    if m.group(0).lower() != "environment":
        return False
    before = re.findall(r"[a-z]+", text[max(0, m.start() - 40):m.start()].lower())
    after = text[m.end():m.end() + 20].lower()
    if before and before[-1] in _BUSINESS_MODIFIERS:
        return True
    return bool(re.match(r"\s+in which\b", after))


def environmental_hits(sentence: str) -> list[str]:
    """Environmental keywords found in `sentence` (after exclusions)."""
    hits = []
    fin_spans = [m.span() for m in _FIN_SUSTAINABILITY.finditer(sentence)]
    fin_spans += [m.span() for m in _NON_ENV_PHRASES.finditer(sentence)]
    for m in KEYWORD_RE.finditer(sentence):
        if _is_business_environment(sentence, m):
            continue
        if any(s <= m.start() < e for s, e in fin_spans):
            continue
        hits.append(m.group(0).lower())
    return hits


def is_environmental(sentence: str) -> bool:
    return bool(environmental_hits(sentence))
