#!/usr/bin/env python3
"""data/ 配下の文書を Qdrant に登録する。

  python ingest.py                # 差分登録：変わったファイルだけ登録し直す
  python ingest.py --force        # 全ファイルを登録し直す（古いチャンクは消してから入れる）
  python ingest.py --recreate     # コレクションと台帳を作り直して全件登録
  python ingest.py --delete FILE  # 指定ファイル由来のチャンクを削除

「変わった」は、台帳（state/ledger.sqlite3）に記録した内容のハッシュ・分割の設定・
埋め込みのバックエンドのどれかが違うこと。分割の設定（CHUNK_SIZE など）を変えると、
内容が同じでも全ファイルが登録し直しの対象になる。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core import chunker, config, ledger, store, validity


def load_documents() -> list[tuple[str, str, str]]:
    """(source, category, text) の列を返す。

    data/<カテゴリ>/<ファイル> という構成を前提にする。
    EXTRA_DATA_DIR が設定されていれば、そのフォルダも同じ構成として読む。
    """
    roots = [config.DATA_DIR]
    if config.EXTRA_DATA_DIR:
        roots.append(Path(config.EXTRA_DATA_DIR))
    docs = []
    for root in roots:
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(root)
            category = rel.parts[0] if len(rel.parts) > 1 else "その他"
            docs.append((path.name, category, path.read_text(encoding="utf-8")))
    return docs


def register(source: str, category: str, text: str, force: bool) -> str:
    """1ファイルを登録し、結果を1行の文字列で返す。"""
    chunks = chunker.make_chunks(text)
    n = len(chunks)
    valid = validity.parse(text)
    period = f"  適用期間 {validity.fmt(*valid)}" if any(valid) else ""

    if force:
        store.delete_by_source(source)
        store.upsert_chunks(source, category, chunks, valid=valid)
        ledger.record(source, category, text, n)
        return f"{n:>3} チャンク（登録し直し）{period}"

    if not ledger.needs_update(source, text):
        return "変更なし"

    before = store.existing_indices(source)
    if config.BUG_ORPHAN_CHUNKS:
        # 本番の旧実装：足りない番号を足すだけ。
        # 既にある番号は内容が古くても上書きせず、余った番号も消さない。
        missing = [i for i in range(n) if i not in before]
        store.upsert_chunks(source, category, chunks, indices=missing, valid=valid)
        ledger.record(source, category, text, n)
        return f"{n:>3} チャンク（追加 {len(missing)} 件、既存 {len(before)} 件はそのまま）{period}"

    # 直した実装：全番号を上書きし、もう存在しない番号を消す。
    store.upsert_chunks(source, category, chunks, valid=valid)
    obsolete = sorted(i for i in before if i >= n)
    if obsolete:
        store.delete_indices_from(source, n)
    ledger.record(source, category, text, n)
    msg = f"{n:>3} チャンク（更新"
    return msg + (f"、古い index {obsolete} を削除）" if obsolete else "）") + period


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="コレクションと台帳を作り直す")
    ap.add_argument("--force", action="store_true", help="全ファイルを登録し直す")
    ap.add_argument("--delete", metavar="FILE", help="指定ファイルのチャンクを削除")
    args = ap.parse_args()

    print(f"backend={config.EMBEDDING_BACKEND} dim={config.vector_size()}"
          f" collection={config.COLLECTION_NAME}  {ledger.chunk_config()}")
    if config.EXTRA_DATA_DIR:
        print(f"追加フォルダ: {config.EXTRA_DATA_DIR}")
    if config.BUG_SPARSE_UNSTABLE_HASH:
        print("!! BUG_SPARSE_UNSTABLE_HASH=1 : 壊れた sparse 実装で登録します")
    if config.BUG_ORPHAN_CHUNKS:
        print("!! BUG_ORPHAN_CHUNKS=1 : 差分登録で古いチャンクを消しません")

    store.ensure_collection(recreate=args.recreate)
    if args.recreate:
        ledger.reset()

    if args.delete:
        store.delete_by_source(args.delete)
        ledger.remove(args.delete)
        print(f"削除: {args.delete}  残り {store.count()} チャンク")
        return 0

    docs = load_documents()
    if not docs:
        print(f"文書が見つかりません: {config.DATA_DIR}", file=sys.stderr)
        return 1

    for source, category, text in docs:
        print(f"  {source:<40} {category:<12} {register(source, category, text, args.force)}")

    orphans = store.delete_orphans({source for source, _, _ in docs})
    for source in orphans:
        ledger.remove(source)
        print(f"  {source:<40} {'(data/に無し)':<12} 削除")

    expected = sum(r["chunk_count"] for r in ledger.all_rows())
    actual = store.count()
    print(f"\n台帳のチャンク数: {expected}")
    print(f"コレクション内の合計: {actual} チャンク")
    if expected != actual:
        print(f"!! 台帳と {actual - expected:+d} チャンクずれています。tools/check_orphans.py で確認してください")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
