"""質問に検索（RAG）が要るかを判定する。

「ありがとう」「なるほど」のような相槌に、毎回社内文書を検索して答えるのは無駄で、
的外れな文書を拾う原因にもなる。そこで、相槌には検索しない。

本番では Haiku（expand_query）がこの判定もしていた。サンプルでは、
APIキーが無くても動くよう、相槌の言い回しで判定する簡易版にしている。

「なるほど、じゃあ乾燥ハーブは？」のように相槌の後に質問が続く場合は、検索する。
相槌だけの質問（「なるほど、そうなんですね」）を検索不要と判定したあと、
どのプロンプトへ回すかで、事件簿③の結果が変わる（core/answer.py）。
"""
from __future__ import annotations

import re

_AIZUCHI = (
    "ありがとう",
    "ありがとうございます",
    "なるほど",
    "了解",
    "りょうかい",
    "わかりました",
    "分かりました",
    "そうなんだ",
    "そうなんですね",
    "はい",
    "OK",
    "ok",
    "助かります",
    "すごい",
    "そうですか",
    "そうなんですか",
    "ね",
    "です",
    "ございます",
    "よろしくお願いします",
)


_PUNCT = re.compile(r"^[\s　、。,.！!？?〜ー…]+")


def is_rag_needed(question: str) -> bool:
    """相槌だけでできた質問は検索不要と判定する。"""
    q = _PUNCT.sub("", question)
    while q:
        word = next((w for w in sorted(_AIZUCHI, key=len, reverse=True) if q.startswith(w)), None)
        if word is None:
            return True  # 相槌以外の言葉が残る＝中身のある質問
        q = _PUNCT.sub("", q[len(word):])
    return False
