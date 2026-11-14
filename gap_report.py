#!/usr/bin/env python3
"""質問ログから「答えられなかった質問」を抜き出す。

  python gap_report.py                 # logs/questions.jsonl を集計
  python gap_report.py path/to/log.jsonl

第6章「質問ログから穴を見つける」。検索が0件だった質問は、
文書が無いか、あっても言葉が合っていない。どちらも改善の入口になる。
追加質問（state=continuation）も取りこぼさないよう、state ではなく件数で見る。
ただし検索していない質問（相槌の state=free、構造化データで答えた state=structured）は数えない。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from core import config


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else config.QUESTION_LOG)
    if not path.exists():
        print(f"ログがありません: {path}")
        return 1

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    states = Counter(r["state"] for r in records)
    nohit = Counter(r["question"] for r in records if r["hits"] == 0 and r["state"] not in ("free", "structured"))
    truncated = [r for r in records if r.get("truncated")]

    print(f"{path}  {len(records)} 件  " + "  ".join(f"{k}={v}" for k, v in states.most_common()))

    print(f"\n■ 検索が0件だった質問 {sum(nohit.values())} 件")
    for question, n in nohit.most_common():
        print(f"  {n:>3}  {question}")

    print(f"\n■ 回答が途中で切れた質問 {len(truncated)} 件")
    for r in truncated:
        print(f"  {r['ts']}  {r['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
