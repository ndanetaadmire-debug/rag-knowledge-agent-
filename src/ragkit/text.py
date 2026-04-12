from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "get",
        "gets",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def stem(t: str) -> str:
    """Tiny suffix stripper: enough to match 'refunds'/'refunded'/'refunding' without a dependency."""
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text)]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip()) if s.strip()]


# Query rewriting: users say "broken", policies say "damaged". In production this is an LLM rewrite
# step or a learned synonym table; a small curated map keeps the default deterministic and testable.
SYNONYMS: dict[str, frozenset[str]] = {
    "broken": frozenset({"damag", "defect"}),
    "damag": frozenset({"broken", "defect"}),
    "minimum": frozenset({"least"}),
    "length": frozenset({"character", "long"}),
    "parcel": frozenset({"order", "package"}),
    "package": frozenset({"order", "parcel"}),
    "wore": frozenset({"worn"}),
    "worn": frozenset({"wore"}),
    "cost": frozenset({"price", "fee"}),
    "free": frozenset({"no cost"}),
    "money": frozenset({"refund"}),
}


def expand(term_list: list[str]) -> list[set[str]]:
    """Each query term becomes a set: itself plus its synonyms (stemmed)."""
    return [{t, *(stem(s) for s in SYNONYMS.get(t, frozenset()))} for t in term_list]
