"""ragkit ask "question" [--groups hr,staff] | ragkit eval qa.json  (data dir via --data)"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragkit.answer import ask
from ragkit.corpus import load_directory
from ragkit.evaluate import evaluate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ragkit")
    p.add_argument("--data", type=Path, default=Path("data/handbook"))
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--groups", default="")
    e = sub.add_parser("eval")
    e.add_argument("qa", type=Path)
    e.add_argument("--min-recall", type=float, default=0.9)
    e.add_argument("--min-refusal", type=float, default=0.9)
    args = p.parse_args(argv)
    index = load_directory(args.data)

    if args.cmd == "ask":
        ans = ask(index, args.question, groups=frozenset(g for g in args.groups.split(",") if g))
        print(ans.text)
        for c in ans.citations:
            print(f"  [{c.n}] {c.title} > {c.section}")
        return 0
    r = evaluate(index, json.loads(args.qa.read_text()))
    print(f"recall@4={r.recall_at_k:.0%}  MRR={r.mrr:.2f}  refusal-accuracy={r.refusal_accuracy:.0%}  n={r.n}")
    for m in r.misses:
        print(f"  miss: {m}")
    return 0 if r.recall_at_k >= args.min_recall and r.refusal_accuracy >= args.min_refusal else 1


if __name__ == "__main__":
    sys.exit(main())
