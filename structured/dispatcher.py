"""構造化データ層：数える・並べる質問を、RAG ではなく表から答える。

RAG は「検索して、上位のチャンクを渡す」仕組みなので、
「会議メモは全部で何件」「拠点はいくつ」のように、文書の外にある事実や
件数は答えられない（第5章 事件簿⑤）。そういう質問は、ここで先に拾う。

  質問 → dispatch() → registry.json の意図に当たれば、ハンドラが表を引いて答える
                     → 当たらなければ None（いつもの RAG へ）

本番では Haiku が意図を判定していた（dispatcher.py ＋ *.intent.md）。サンプルでは、
APIキーが無くても動くよう、registry.json のキーワードで判定する簡易版にしている。

表：
  documents : 登録台帳（state/ledger.sqlite3）。どのカテゴリに何の文書があるか
  sites     : 拠点の一覧（structured/sites.csv）。人が管理するマスタ
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from core import config, ledger

_HERE = Path(__file__).resolve().parent


@dataclass
class StructuredAnswer:
    intent: str
    text: str  # そのまま利用者に返せる回答文
    sql: str  # 引いた SQL（説明・デバッグ用）
    rows: list[dict] = field(default_factory=list)


def _registry() -> list[dict]:
    return json.loads((_HERE / "registry.json").read_text(encoding="utf-8"))["intents"]


def match_intent(question: str) -> str | None:
    """registry.json の意図に照合する。subject と ask の両方の語を含めば当たり。"""
    for intent in _registry():
        if any(w in question for w in intent["subject"]) and any(w in question for w in intent["ask"]):
            return intent["name"]
    return None


def _db() -> sqlite3.Connection:
    """台帳と拠点マスタを、1つのメモリ上の SQLite にまとめる。"""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE documents (source TEXT, category TEXT)")
    db.executemany(
        "INSERT INTO documents VALUES (?, ?)",
        [(r["source"], r["category"]) for r in ledger.all_rows()],
    )
    db.execute(
        "CREATE TABLE sites (name TEXT, role TEXT, resident_staff INTEGER, dorm_capacity INTEGER, note TEXT)"
    )
    with open(_HERE / "sites.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    db.executemany(
        "INSERT INTO sites VALUES (?, ?, ?, ?, ?)",
        [
            (
                r["name"],
                r["role"],
                int(r["resident_staff"]) if r["resident_staff"] else None,
                int(r["dorm_capacity"]) if r["dorm_capacity"] else None,
                r["note"],
            )
            for r in rows
        ],
    )
    return db


def _doc_count(question: str, db: sqlite3.Connection) -> StructuredAnswer:
    categories = [r[0] for r in db.execute("SELECT DISTINCT category FROM documents")]
    category = next((c for c in categories if c in question), None)
    if category:
        sql = "SELECT source FROM documents WHERE category = ? ORDER BY source"
        rows = [dict(r) for r in db.execute(sql, (category,))]
        names = "、".join(r["source"] for r in rows)
        text = f"{category}は全部で{len(rows)}件です（{names}）。"
    else:
        sql = "SELECT category, COUNT(*) AS n FROM documents GROUP BY category ORDER BY category"
        rows = [dict(r) for r in db.execute(sql)]
        total = sum(r["n"] for r in rows)
        detail = "、".join(f"{r['category']} {r['n']}件" for r in rows)
        text = f"登録されている文書は全部で{total}件です（{detail}）。"
    if not rows:
        text = "登録台帳が空です。先に python ingest.py で文書を登録してください。"
    return StructuredAnswer("doc_count", text, sql, rows)


def _site_count(question: str, db: sqlite3.Connection) -> StructuredAnswer:
    sql = "SELECT name, role FROM sites ORDER BY rowid"
    rows = [dict(r) for r in db.execute(sql)]
    detail = "、".join(f"{r['name']}（{r['role']}）" for r in rows)
    return StructuredAnswer("site_count", f"拠点は{len(rows)}か所です。{detail}。", sql, rows)


def _site_staff(question: str, db: sqlite3.Connection) -> StructuredAnswer:
    names = [r[0] for r in db.execute("SELECT name FROM sites")]
    targets = [n for n in names if n in question] or names
    sql = "SELECT name, resident_staff, dorm_capacity FROM sites WHERE name IN (%s)" % ",".join(
        "?" * len(targets)
    )
    rows = [dict(r) for r in db.execute(sql, targets)]
    parts = []
    for r in rows:
        if r["resident_staff"] is not None:
            parts.append(f"{r['name']}拠点の常駐は{r['resident_staff']}名です。")
        else:
            # 書いていない数字は、表にも無い。近くにある別の数字で埋めない。
            extra = f"（社員寮の定員は{r['dorm_capacity']}名ですが、在籍人数ではありません）" if r["dorm_capacity"] else ""
            parts.append(f"{r['name']}拠点の在籍人数のデータはありません{extra}。")
    return StructuredAnswer("site_staff", "".join(parts), sql, rows)


_HANDLERS = {"doc_count": _doc_count, "site_count": _site_count, "site_staff": _site_staff}


def dispatch(question: str) -> StructuredAnswer | None:
    """構造化データで答えられる質問なら答えを返す。それ以外は None（RAG へ）。"""
    if not config.STRUCTURED:
        return None
    intent = match_intent(re.sub(r"\s+", "", question))
    if intent is None:
        return None
    return _HANDLERS[intent](question, _db())
