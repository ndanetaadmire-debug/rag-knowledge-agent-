from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from ragkit.answer import ask
from ragkit.corpus import load_directory
from ragkit.retrieval import HybridIndex


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def create_app(index: HybridIndex | None = None) -> FastAPI:
    idx = index or load_directory(Path(os.getenv("RAG_DATA_DIR", "data/handbook")))
    app = FastAPI(title="RAG knowledge agent")

    @app.get("/healthz")
    def health() -> dict[str, object]:
        return {"status": "ok", "chunks": len(idx.chunks)}

    @app.post("/ask")
    def ask_endpoint(body: Question, x_user_groups: Annotated[str, Header()] = "") -> dict[str, object]:
        # Groups come from the verified identity token in production, never from the client.
        groups = frozenset(g.strip() for g in x_user_groups.split(",") if g.strip())
        return asdict(ask(idx, body.question, groups=groups))

    return app
