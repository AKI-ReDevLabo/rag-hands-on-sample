#!/usr/bin/env python3
"""ファイルをチャンクに切って中身を表示する（登録はしない）。

  python chunks.py data/取引先対応/季節のお便り_2026秋.md
  python chunks.py FILE --raw                  # 正規化せずに切る
  CHUNK_SIZE=1000 CHUNK_OVERLAP=0 MIN_CHUNK_SIZE=0 python chunks.py FILE

ingest.py は Path.read_text() で読むため、CR+LF は読み込み時に LF へ
変換されて見えなくなる。ここではバイト列から読んで CR をそのまま残し、
行末の CR を「␍」として表示する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core import chunker, config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--raw", action="store_true", help="正規化（改行・空行・記号行）をしない")
    args = ap.parse_args()

    text = Path(args.file).read_bytes().decode("utf-8")
    chunks = chunker.make_chunks(text, normalize_text=not args.raw)

    print(
        f"{args.file}  {len(text)} 文字  CR+LF={text.count(chr(13) + chr(10))} 行"
        f"  normalize={'off' if args.raw else 'on'}"
    )
    print(
        f"CHUNK_SIZE={config.CHUNK_SIZE} CHUNK_OVERLAP={config.CHUNK_OVERLAP}"
        f" MIN_CHUNK_SIZE={config.MIN_CHUNK_SIZE}  → {len(chunks)} チャンク"
    )
    for i, chunk in enumerate(chunks, 1):
        print(f"\n----- [{i}] {len(chunk)} 文字 -----")
        print(chunk.replace("\r", "␍"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
