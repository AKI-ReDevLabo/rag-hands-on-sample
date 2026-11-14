"""管理番号のようなコードを、本文や質問から取り出す。

「KR-3104-01」のようなコードは意味を持たない文字列なので、dense では近さを測れない。
sparse でも、形態素解析が「2609」「-」「01」に砕いてしまう。
どちらのベクトルにも乗らないので、正規表現で取り出して payload の完全一致で探す。

  登録時：チャンクの本文からコードを取り出し、payload の codes に入れる
  検索時：質問からコードを取り出せたら、codes で絞り込んで検索する（コード指定モード）

本番では質問側の取り出しを Haiku（expand_query）が担っていた。サンプルは正規表現だけ。
"""
from __future__ import annotations

import re

from . import config

# 管理番号：前4桁＝年月、後2桁＝連番。
# 後ろに「-」や数字が続くもの（2026-09-14 のような日付）は除く。
CODE_RE = re.compile(r"\b[A-Z]{2}-\d{4}-\d{2}(?![-\d])")

# 旧い正規表現。日付「2026-09-14」の先頭「2026-09」まで管理番号として拾ってしまう。
_BUGGY_CODE_RE = re.compile(r"\b\d{4}-\d{2}")


def extract_codes(text: str) -> list[str]:
    pattern = _BUGGY_CODE_RE if config.BUG_CODE_REGEX_MATCHES_DATE else CODE_RE
    return sorted(set(pattern.findall(text)))
