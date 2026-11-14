"""チャンク分割。

本番で踏んだ地雷を踏まえた実装：
  - CR+LF を正規化する（Windows 由来ファイルの ^M 対策）
  - 連続空行を圧縮する（空行でチャンクが不必要に切れるのを防ぐ）
  - 記号だけの行を落とす（``` や --- がチャンクを汚す）
  - 空行で段落に分け、CHUNK_SIZE を超える段落だけを固定長で切る
  - 短すぎるチャンクを次に併合する（収束するまで繰り返す。CHUNK_SIZE を超えない範囲で）
"""
from __future__ import annotations

import re

from . import config

# 記号だけで構成された行（``` や --- や === など）
_NOISE_LINE = re.compile(r"^\s*(?:`{3,}|-{3,}|={3,}|\*{3,}|_{3,})\s*$")


def normalize(text: str) -> str:
    """改行コードと空行を正規化する。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if not _NOISE_LINE.match(ln)]
    text = "\n".join(lines)
    # 3行以上の連続空行を2行に圧縮
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """固定長＋オーバーラップで切る。"""
    if size <= overlap:
        raise ValueError("CHUNK_SIZE は CHUNK_OVERLAP より大きい必要があります")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk.strip())
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def merge_short_chunks(chunks: list[str], min_size: int, max_size: int | None = None) -> list[str]:
    """min_size 未満のチャンクを、次のチャンクに前方併合する（収束するまで繰り返す）。

    次が無い（末尾の）短いチャンクは、直前のチャンクに後方併合する。
    併合後の長さが max_size（既定は CHUNK_SIZE）を超える場合は併合しない。

    BUG_MERGE_NO_LIMIT=1 では、この上限チェックをしない（上限チェックを足す前の実装）。
    短い見出しや表が、隣の段落と長さを問わず結合され、CHUNK_SIZE を超えるチャンクができる。
    """
    limit = None if config.BUG_MERGE_NO_LIMIT else (max_size or config.CHUNK_SIZE)

    def fits(a: str, b: str) -> bool:
        return limit is None or len(a) + len(b) + 1 <= limit

    while True:
        merged: list[str] = []
        changed = False
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            if len(cur) < min_size and i + 1 < len(chunks) and fits(cur, chunks[i + 1]):
                merged.append(cur + "\n" + chunks[i + 1])  # 前方併合
                i += 2
                changed = True
            elif len(cur) < min_size and merged and fits(merged[-1], cur):
                merged[-1] += "\n" + cur  # 次が無い（または次とは入らない）ので後方併合
                i += 1
                changed = True
            else:
                merged.append(cur)
                i += 1
        chunks = merged
        if not changed:
            return chunks


def split_paragraphs(text: str) -> list[str]:
    """空行で段落に分ける。

    正規化していない CR+LF のテキストでは空行が「\\r\\n\\r\\n」になり、
    "\\n\\n" で区切れないので、ファイル全体が1段落になる（段落0）。
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def make_chunks(text: str, normalize_text: bool = True) -> list[str]:
    """本文テキストをチャンクのリストにする。

    normalize_text=False は正規化前の切れ方を見るためのもの（chunks.py --raw）。
    """
    if normalize_text:
        text = normalize(text)
    if not text:
        return []
    chunks: list[str] = []
    for para in split_paragraphs(text):
        if len(para) <= config.CHUNK_SIZE:
            chunks.append(para)
        else:
            # 長い段落だけ、固定長＋オーバーラップで切る
            chunks.extend(_split_fixed(para, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
    return merge_short_chunks(chunks, config.MIN_CHUNK_SIZE)
