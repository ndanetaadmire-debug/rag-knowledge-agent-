"""Hybrid retrieval: BM25 (exact terms, codes, names) + dense vectors (paraphrase), fused with
Reciprocal Rank Fusion, diversified with MMR, and filtered by the caller's permissions *before*
ranking, so restricted text can never be retrieved, quoted or leaked into a prompt."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ragkit.chunking import Chunk
from ragkit.text import expand, terms


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Offline, deterministic embedder (feature-hashed word + character trigrams, L2-normalised).
    Good enough to demo and test hybrid retrieval; swap for a real model behind the same port."""

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        toks = terms(text)
        feats = toks + [f"#{t[i : i + 3]}" for t in toks for i in range(max(1, len(t) - 2))]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dims] += 1.0 if (h >> 63) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridIndex:
    def __init__(self, embedder: Embedder | None = None, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()
        self._vecs: list[list[float]] = []

    def add(self, chunks: Iterable[Chunk]) -> None:
        new = [c for c in chunks if c.id not in {x.id for x in self.chunks}]  # idempotent re-ingest
        vecs = self.embedder.embed([c.context for c in new])
        for c, v in zip(new, vecs, strict=True):
            tf = Counter(terms(c.context))
            self.chunks.append(c)
            self._tf.append(tf)
            self._len.append(sum(tf.values()))
            self._df.update(tf.keys())
            self._vecs.append(v)

    def _bm25(self, q: list[str], idx: Sequence[int]) -> list[tuple[int, float]]:
        n = len(self.chunks)
        avg = sum(self._len) / max(1, n)
        out = []
        for i in idx:
            s = 0.0
            for t in q:
                f = self._tf[i].get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / avg))
            if s > 0:
                out.append((i, s))
        return sorted(out, key=lambda x: -x[1])

    def search(
        self,
        query: str,
        *,
        groups: frozenset[str],
        k: int = 5,
        candidates: int = 20,
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
    ) -> list[Hit]:
        # 1. Permission filter first. Public chunks have no groups.
        visible = [i for i, c in enumerate(self.chunks) if not c.allowed_groups or c.allowed_groups & groups]
        if not visible:
            return []
        qv = self.embedder.embed([query])[0]
        q_terms = sorted(set().union(*expand(terms(query)))) if terms(query) else []
        bm = self._bm25(q_terms, visible)[:candidates]
        dense = sorted(((i, cosine(qv, self._vecs[i])) for i in visible), key=lambda x: -x[1])[:candidates]

        # 2. Reciprocal Rank Fusion: robust to the two scorers' incomparable scales.
        fused: dict[int, float] = {}
        bm_rank = {i: r for r, (i, _) in enumerate(bm)}
        de_rank = {i: r for r, (i, s) in enumerate(dense) if s > 0}
        for ranks in (bm_rank, de_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        pool = sorted(fused, key=lambda i: -fused[i])

        # 3. MMR: prefer relevant chunks that add something new (avoid five near-duplicates).
        chosen: list[int] = []
        while pool and len(chosen) < k:
            best = max(
                pool,
                key=lambda i: (
                    mmr_lambda * fused[i] * 60
                    - (1 - mmr_lambda) * max((cosine(self._vecs[i], self._vecs[j]) for j in chosen), default=0.0)
                ),
            )
            chosen.append(best)
            pool.remove(best)
        return [Hit(self.chunks[i], round(fused[i], 5), bm_rank.get(i), de_rank.get(i)) for i in chosen]
