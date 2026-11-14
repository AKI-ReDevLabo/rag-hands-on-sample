"""追加質問を、検索用の一文に書き直す。

会話を続けると「それの締切は？」「さっきのは返品できる？」のように言葉が省略される。
そのまま検索すると「それ」「締切」だけで探すことになり、狙った文書を引けない。
直前のやりとりを Haiku に渡して、省略を補った一文にしてから検索する。

書き直しに失敗したとき（API の失敗、max_tokens での打ち切り、空の返答）は、
元の質問のまま検索する。黙って戻らないよう、理由を note に入れる。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Expansion:
    query: str  # 検索に使う文
    expanded: bool  # 書き直しが走ったか
    note: str | None = None  # 書き直さなかった理由（初回の質問など、理由が無いときは None）


def build_expand_prompt(question: str, history: list[dict]) -> str:
    lines = []
    for t in history:
        who = "質問" if t["role"] == "user" else "回答"
        lines.append(f"{who}: {t['content'][:300]}")
    convo = "\n".join(lines)
    return f"""次の会話の続きの質問を、社内文書を検索するための一文に書き直してください。

直前の会話:
{convo}

続きの質問: {question}

書き直しのルール:
- 「それ」「さっきの」などの指示語や省略を、直前の会話の言葉で補う
- 品目名・拠点名・日付など固有名詞は、会話に出てきた表記のまま使う
- 質問に答えない。検索用の一文だけを出力する（前置き・引用符・説明は禁止）
- 続きの質問が単独で意味の通る文なら、そのまま出力する"""


def _clean(raw: str) -> str:
    text = re.sub(r"^```\w*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    text = text.splitlines()[0].strip() if text else ""
    return text.strip("「」\"'　 ")


def expand_query(question: str, history: list[dict] | None, client=None) -> Expansion:
    """history は [{"role": "user"|"assistant", "content": ...}] の列。"""
    turns = list(history or [])[-config.EXPAND_HISTORY_MESSAGES :]
    if not turns:
        return Expansion(query=question, expanded=False)  # 最初の質問は書き直さない
    if not config.EXPAND_QUERY:
        return Expansion(query=question, expanded=False)
    if not config.ANTHROPIC_API_KEY and client is None:
        return Expansion(query=question, expanded=False, note="ANTHROPIC_API_KEY が設定されていません")

    import anthropic

    client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        resp = client.messages.create(
            model=config.EXPAND_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": build_expand_prompt(question, turns)}],
        )
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
        return _fallback(question, f"追加質問の書き直しに失敗しました（{type(e).__name__}）")

    if resp.stop_reason == "max_tokens":
        return _fallback(question, "追加質問の書き直しが max_tokens で途中で切れました")

    text = _clean("".join(b.text for b in resp.content if b.type == "text"))
    if not text:
        return _fallback(question, "追加質問の書き直しが空でした")
    return Expansion(query=text, expanded=True)


def _fallback(question: str, reason: str) -> Expansion:
    logger.warning("元の質問のまま検索します: %s", reason)
    return Expansion(query=question, expanded=False, note=reason)
