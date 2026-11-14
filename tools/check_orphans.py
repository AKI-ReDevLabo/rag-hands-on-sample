#!/usr/bin/env python3
"""台帳と Qdrant の実体を突き合わせて、孤児チャンクを探す（登録はしない）。

  python tools/check_orphans.py

ファイルごとに次の3つを数える。
  孤児     : 台帳のチャンク数より大きい chunk_index が Qdrant に残っている
  欠け     : 台帳のチャンク数の範囲にあるはずの chunk_index が無い
  古い内容 : 今のファイルを今の設定で切り直した結果と、payload の本文が違う

台帳は「登録した後に書く記録」なので、台帳だけを見ても実体のずれは分からない。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import chunker, config, ledger, store  # noqa: E402


def payload_texts(source: str) -> dict[int, str]:
    texts: dict[int, str] = {}
    offset = None
    while True:
        points, offset = store.client().scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=store._source_filter(source),
            limit=256,
            offset=offset,
            with_payload=["chunk_index", "text"],
            with_vectors=False,
        )
        for p in points:
            texts[p.payload.get("chunk_index")] = p.payload.get("text", "")
        if offset is None:
            return texts


def main() -> int:
    files = {p.name: p for p in config.DATA_DIR.rglob("*.md")}
    rows = ledger.all_rows()
    bad = 0
    total_orphans = 0
    for row in rows:
        source, n = row["source"], row["chunk_count"]
        texts = payload_texts(source)
        orphans = sorted(i for i in texts if i >= n)
        missing = sorted(i for i in range(n) if i not in texts)
        stale = []
        if source in files:
            current = chunker.make_chunks(files[source].read_text(encoding="utf-8"))
            stale = sorted(i for i, t in texts.items() if i < len(current) and t != current[i])
        if orphans or missing or stale:
            bad += 1
            total_orphans += len(orphans)
            parts = []
            if orphans:
                parts.append(f"孤児 index {orphans}")
            if missing:
                parts.append(f"欠け index {missing}")
            if stale:
                parts.append(f"古い内容 index {stale}")
            print(f"  {source:<40} 台帳 {n} / 実体 {len(texts)}  " + "、".join(parts))

    untracked = sorted(store.list_sources() - {r["source"] for r in rows})
    for source in untracked:
        print(f"  {source:<40} 台帳に無いのに Qdrant に残っている")

    print(f"\n{len(rows)} ファイル中 {bad} ファイルにずれ（孤児チャンク {total_orphans} 件）"
          + (f"、台帳に無いファイル {len(untracked)} 件" if untracked else ""))
    if bad or untracked:
        print("直すには: python ingest.py --force")
    return 1 if bad or untracked else 0


if __name__ == "__main__":
    raise SystemExit(main())
