from __future__ import annotations

import json
from pathlib import Path

from ragkit.chunking import chunk_markdown
from ragkit.retrieval import HybridIndex


def load_directory(path: Path, index: HybridIndex | None = None) -> HybridIndex:
    """Load *.md files. An optional `acl.json` maps file stem → list of groups allowed to read it."""
    index = index or HybridIndex()
    acl_file = path / "acl.json"
    acl: dict[str, list[str]] = json.loads(acl_file.read_text()) if acl_file.exists() else {}
    for f in sorted(path.glob("*.md")):
        index.add(chunk_markdown(f.stem, f.read_text(), allowed_groups=frozenset(acl.get(f.stem, []))))
    return index
