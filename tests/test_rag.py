import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragkit.answer import REFUSAL, ask, verify_grounding
from ragkit.api import create_app
from ragkit.chunking import chunk_markdown
from ragkit.cli import main
from ragkit.corpus import load_directory
from ragkit.evaluate import evaluate
from ragkit.retrieval import HashingEmbedder, HybridIndex, cosine
from ragkit.text import expand, terms

DATA = Path("data/handbook")


@pytest.fixture(scope="module")
def index() -> HybridIndex:
    return load_directory(DATA)


def test_chunking_keeps_heading_path_and_respects_budget() -> None:
    md = "# Handbook\n\n## Refunds\n\n### Damaged\n\n" + "\n\n".join(
        f"Paragraph {i}. " + "word " * 40 for i in range(6)
    )
    chunks = chunk_markdown("hb", md, max_chars=500)
    assert len(chunks) > 1
    assert all(c.heading_path == ("Refunds", "Damaged") and c.title == "Handbook" for c in chunks)
    assert all(len(c.text) <= 500 + 200 for c in chunks)  # budget + capped overlap
    assert len({c.id for c in chunks}) == len(chunks)


def test_reingest_is_idempotent() -> None:
    idx = HybridIndex()
    chunks = chunk_markdown("a", "# A\n\n## S\n\nSome text here.")
    idx.add(chunks)
    idx.add(chunks)
    assert len(idx.chunks) == 1


def test_embedder_is_deterministic_and_semantic_ish() -> None:
    e = HashingEmbedder()
    a, b, c = e.embed(["refund for damaged items", "damaged item refunds", "annual leave accrual"])
    assert e.embed(["refund for damaged items"])[0] == a
    assert cosine(a, b) > cosine(a, c)


def test_query_rewriting() -> None:
    assert "damag" in expand(terms("it arrived broken"))[-1]


def test_permission_filter_runs_before_ranking(index: HybridIndex) -> None:
    q = "How many annual leave days do staff get?"
    assert all(h.chunk.doc_id != "leave" for h in index.search(q, groups=frozenset()))
    assert index.search(q, groups=frozenset({"hr"}))[0].chunk.doc_id == "leave"
    assert ask(index, q, groups=frozenset()).refused  # restricted content is never quoted


def test_cited_grounded_answer(index: HybridIndex) -> None:
    a = ask(index, "Who must approve a refund over R2,000?", groups=frozenset())
    assert not a.refused and a.grounded
    assert "operations manager" in a.text
    assert a.citations and a.citations[0].section == "Damaged items"


def test_refuses_off_topic(index: HybridIndex) -> None:
    a = ask(index, "Who won the rugby world cup?", groups=frozenset())
    assert a.refused and a.text == REFUSAL


def test_grounding_verifier_strips_unsupported_claims(index: HybridIndex) -> None:
    sources = {1: "Orders over R750 ship free within South Africa."}
    assert verify_grounding("Orders over R750 ship free within South Africa. [1]", sources) == []
    bad = verify_grounding("Orders ship free worldwide with overnight drone delivery. [1]", sources)
    assert bad and "drone" in bad[0]
    assert verify_grounding("Uncited claim about pricing.", sources)

    class Hallucinator:
        def generate(self, question: str, sources: list[tuple[int, str]]) -> str:
            return "Refunds are instant and paid in Bitcoin to any wallet. [1]"

    a = ask(index, "How long until my refund is paid back?", groups=frozenset(), generator=Hallucinator())
    assert a.refused and not a.grounded and a.unsupported


def test_retrieval_eval_gate(index: HybridIndex) -> None:
    r = evaluate(index, json.loads(Path("data/qa.json").read_text()))
    assert r.recall_at_k == 1.0 and r.mrr >= 0.9 and r.refusal_accuracy == 1.0, r.misses


def test_api_respects_groups() -> None:
    c = TestClient(create_app(load_directory(DATA)))
    q = {"question": "How many annual leave days do staff get?"}
    assert c.post("/ask", json=q).json()["refused"] is True
    hr = c.post("/ask", json=q, headers={"x-user-groups": "hr"}).json()
    assert hr["refused"] is False and hr["citations"][0]["doc_id"] == "leave"
    assert c.get("/healthz").json()["chunks"] > 5


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "data/qa.json"]) == 0
    assert main(["ask", "Do you ship internationally?"]) == 0
    assert "internationally" in capsys.readouterr().out
