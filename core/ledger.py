"""登録台帳（どのファイルを、どの内容・どの分割設定で、何チャンク登録したか）。

本番の registered_documents テーブルに相当する。SQLite の1ファイルに持つ。

台帳は「登録した後に書く記録」であって、Qdrant の実体そのものではない。
台帳とQdrant がずれていないかは tools/check_orphans.py で確かめる。
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS registered_documents (
    source            TEXT PRIMARY KEY,
    category          TEXT,
    content_hash      TEXT NOT NULL,
    chunk_config      TEXT NOT NULL,
    embedding_backend TEXT NOT NULL,
    chunk_count       INTEGER NOT NULL,
    registered_at     TEXT NOT NULL
)
"""


def _conn() -> sqlite3.Connection:
    path = Path(config.LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_config() -> str:
    """分割の設定。これが変わったら、内容が同じでも登録し直す。"""
    return (
        f"size={config.CHUNK_SIZE},overlap={config.CHUNK_OVERLAP},min={config.MIN_CHUNK_SIZE}"
        f",merge_limit={'off' if config.BUG_MERGE_NO_LIMIT else 'on'}"
    )


def get(source: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM registered_documents WHERE source = ?", (source,)).fetchone()
    return dict(row) if row else None


def all_rows() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM registered_documents ORDER BY source")]


def needs_update(source: str, text: str) -> bool:
    row = get(source)
    return not (
        row
        and row["content_hash"] == content_hash(text)
        and row["chunk_config"] == chunk_config()
        and row["embedding_backend"] == config.EMBEDDING_BACKEND
    )


def record(source: str, category: str, text: str, chunk_count: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO registered_documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source,
                category,
                content_hash(text),
                chunk_config(),
                config.EMBEDDING_BACKEND,
                chunk_count,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def remove(source: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM registered_documents WHERE source = ?", (source,))


def reset() -> None:
    with _conn() as c:
        c.execute("DELETE FROM registered_documents")
