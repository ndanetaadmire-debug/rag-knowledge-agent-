"""Structure-aware chunking.

Splitting every N characters cuts tables and sentences in half and separates a rule from the
heading that gives it meaning. Instead: split by Markdown headings, keep the heading path on every
chunk (so "Refunds > Damaged items" travels with its text), pack whole paragraphs up to a budget,
and overlap one sentence between neighbouring chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ragkit.text import sentences


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    heading_path: tuple[str, ...]
    text: str
    allowed_groups: frozenset[str] = field(default_factory=frozenset)

    @property
    def context(self) -> str:
        """Text that is indexed: heading path + body, so headings are searchable."""
        return " > ".join((self.title, *self.heading_path)) + "\n" + self.text


def chunk_markdown(
    doc_id: str, markdown: str, *, max_chars: int = 900, allowed_groups: frozenset[str] = frozenset()
) -> list[Chunk]:
    title = doc_id
    if m := re.search(r"^#\s+(.+)$", markdown, re.M):
        title = m.group(1).strip()

    sections: list[tuple[tuple[str, ...], str]] = []
    path: list[str] = []
    buf: list[str] = []
    for line in markdown.splitlines():
        h = re.match(r"^(#{2,6})\s+(.+)$", line)
        if h:
            if buf:
                sections.append((tuple(path), "\n".join(buf).strip()))
                buf = []
            level = len(h.group(1)) - 1
            path = [*path[: level - 1], h.group(2).strip()]
        elif not re.match(r"^#\s", line):
            buf.append(line)
    if buf:
        sections.append((tuple(path), "\n".join(buf).strip()))

    chunks: list[Chunk] = []
    for heading_path, body in sections:
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > max_chars:
                chunks.append(_mk(doc_id, title, heading_path, current, allowed_groups))
                current = (_overlap(current) + "\n\n" if _overlap(current) else "") + para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(_mk(doc_id, title, heading_path, current, allowed_groups))
    return chunks


def _mk(doc_id: str, title: str, path: tuple[str, ...], text: str, groups: frozenset[str]) -> Chunk:
    digest = hashlib.sha1(f"{doc_id}|{'/'.join(path)}|{text}".encode()).hexdigest()[:12]
    return Chunk(
        id=f"{doc_id}#{digest}", doc_id=doc_id, title=title, heading_path=path, text=text, allowed_groups=groups
    )


def _overlap(text: str, max_chars: int = 200) -> str:
    """Last sentence of the last paragraph, capped, so neighbouring chunks share context
    without unpunctuated text dragging the whole previous chunk along."""
    last_para = re.split(r"\n\s*\n", text.strip())[-1]
    tail = (sentences(last_para) or [""])[-1]
    return tail if len(tail) <= max_chars else ""
