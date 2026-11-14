#!/usr/bin/env python3
"""Qdrant に登録された管理番号（payload の codes）を、ファイルごとに一覧する。

  python tools/list_codes.py

BUG_CODE_REGEX_MATCHES_DATE=1 で登録すると、日付の一部（2026-09 など）が
管理番号として紛れ込んでいるのが見える。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config, store  # noqa: E402


def main() -> int:
    by_source: dict[str, set[str]] = defaultdict(set)
    offset = None
    while True:
        points, offset = store.client().scroll(
            collection_name=config.COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["source", "codes"],
            with_vectors=False,
        )
        for p in points:
            by_source[p.payload.get("source")].update(p.payload.get("codes") or [])
        if offset is None:
            break
    for source in sorted(s for s in by_source if by_source[s]):
        print(f"  {source:<32} {', '.join(sorted(by_source[source]))}")
    if not any(by_source.values()):
        print("  管理番号を持つチャンクはありません")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
