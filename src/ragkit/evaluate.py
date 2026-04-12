"""Retrieval evaluation: recall@k and MRR over a labelled question set, plus refusal accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragkit.answer import ask
from ragkit.retrieval import HybridIndex


@dataclass(frozen=True, slots=True)
class EvalReport:
    recall_at_k: float
    mrr: float
    refusal_accuracy: float
    n: int
    misses: list[str]


def evaluate(index: HybridIndex, qa: list[dict[str, Any]], *, k: int = 4) -> EvalReport:
    recall, rr, refusal_ok, misses = 0.0, 0.0, 0, []
    answerable = [q for q in qa if q.get("expected_doc")]
    for q in qa:
        groups = frozenset(q.get("groups", []))
        if not q.get("expected_doc"):
            refusal_ok += ask(index, q["question"], groups=groups).refused
            continue
        hits = index.search(q["question"], groups=groups, k=k)
        rank = next(
            (
                r
                for r, h in enumerate(hits, 1)
                if h.chunk.doc_id == q["expected_doc"]
                and (not q.get("expected_section") or q["expected_section"] in h.chunk.heading_path)
            ),
            None,
        )
        if rank:
            recall += 1
            rr += 1 / rank
        else:
            misses.append(q["question"])
        refusal_ok += not ask(index, q["question"], groups=groups).refused
    n = len(answerable)
    return EvalReport(
        round(recall / max(1, n), 3), round(rr / max(1, n), 3), round(refusal_ok / max(1, len(qa)), 3), len(qa), misses
    )
