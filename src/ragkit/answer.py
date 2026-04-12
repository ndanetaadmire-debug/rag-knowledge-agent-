"""Answer generation with citations, groundedness verification and principled refusal."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ragkit.retrieval import Hit, HybridIndex
from ragkit.text import expand, sentences, terms


@dataclass(frozen=True, slots=True)
class Citation:
    n: int
    doc_id: str
    title: str
    section: str
    chunk_id: str


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    refused: bool = False
    reason: str = ""
    unsupported: list[str] = field(default_factory=list)


class Generator(Protocol):
    """LLM port. Must answer only from the numbered sources and cite them like [1]."""

    def generate(self, question: str, sources: list[tuple[int, str]]) -> str: ...


SYSTEM_PROMPT = (
    "Answer using ONLY the numbered sources. Cite every sentence with its source number like [2]. "
    "If the sources do not contain the answer, reply exactly: I don't know based on the available documents."
)


class ExtractiveGenerator:
    """Offline generator: picks the sentences that best answer the question and cites them.
    Deterministic, so tests and CI don't need an API key. Replace with an LLM behind the same port."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, sources: list[tuple[int, str]]) -> str:
        q = set().union(*expand(terms(question))) if terms(question) else set()
        scored = []
        for n, text in sources:
            for s in sentences(text):
                overlap = len(q & set(terms(s)))
                if overlap:
                    scored.append((overlap, -n, s, n))
        if not scored:
            return "I don't know based on the available documents."
        top = max(o for o, *_ in scored)
        # Keep only sentences nearly as relevant as the best one; a single shared word is not relevance.
        best = sorted((x for x in scored if x[0] * 3 >= top * 2), reverse=True)[: self.max_sentences]
        best.sort(key=lambda x: -x[1])  # keep source order for readability
        return " ".join(f"{s.rstrip()} [{n}]" for _, _, s, n in best)


_CITE = re.compile(r"\[(\d+)\]")
REFUSAL = "I don't know based on the available documents."


def coverage(question: str, hit: Hit) -> float:
    """Share of the question's content terms (or their synonyms) present in the chunk."""
    groups = expand(list(dict.fromkeys(terms(question))))
    doc = set(terms(hit.chunk.context))
    return sum(bool(g & doc) for g in groups) / max(1, len(groups))


def verify_grounding(answer: str, sources: dict[int, str], threshold: float = 0.6) -> list[str]:
    """Each sentence must cite a real source, and most of its content words must appear in that source.
    Returns the unsupported sentences. Cheap, deterministic, and catches the common failure:
    a fluent sentence with a citation that doesn't actually say that."""
    unsupported = []
    for s in sentences(answer):
        cited = [int(n) for n in _CITE.findall(s)]
        body = set(terms(_CITE.sub("", s)))
        if not body:
            continue
        evidence = set().union(*(set(terms(sources[n])) for n in cited if n in sources)) if cited else set()
        if not cited or len(body & evidence) / len(body) < threshold:
            unsupported.append(s)
    return unsupported


def ask(
    index: HybridIndex,
    question: str,
    *,
    groups: frozenset[str],
    generator: Generator | None = None,
    k: int = 4,
    min_coverage: float = 0.5,
) -> Answer:
    hits = index.search(question, groups=groups, k=k)
    # Refuse when retrieval is weak: answering from irrelevant context is how RAG hallucinates.
    if not hits or max(coverage(question, h) for h in hits) < min_coverage:
        return Answer(text=REFUSAL, grounded=True, refused=True, reason="insufficient evidence")

    numbered = list(enumerate(hits, start=1))
    sources = {n: h.chunk.text for n, h in numbered}
    text = (generator or ExtractiveGenerator()).generate(question, list(sources.items()))
    if text.strip() == REFUSAL:
        return Answer(text=REFUSAL, refused=True, reason="generator declined")

    used = {int(n) for n in _CITE.findall(text)}
    citations = [
        Citation(n, h.chunk.doc_id, h.chunk.title, " > ".join(h.chunk.heading_path), h.chunk.id)
        for n, h in numbered
        if n in used
    ]
    unsupported = verify_grounding(text, sources)
    if unsupported:
        # Fail safe: strip unsupported sentences; if nothing is left, refuse.
        kept = [s for s in sentences(text) if s not in unsupported]
        if not kept:
            return Answer(
                text=REFUSAL,
                refused=True,
                grounded=False,
                reason="answer not supported by sources",
                unsupported=unsupported,
            )
        text = " ".join(kept)
    return Answer(text=text, citations=citations, grounded=not unsupported, unsupported=unsupported)
