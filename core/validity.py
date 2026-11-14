"""文書の適用期間（いつからいつまで有効か）を扱う。

規程や出荷規格は改定される。旧版が残っていると、検索は新旧の両方を拾い、
LLM はどちらの数字を答えるか決められない（地雷⑥と同じ構図）。
そこで登録時に、本文の先頭付近にある次の行から適用期間を読み取り、payload に入れる。

  適用期間：2025-04-01〜2026-03-31
  適用開始：2026-04-01
  適用終了：2026-03-31

検索時は、基準日（既定は今日、AS_OF で変更可）に有効な文書だけに絞る。
書いていない文書は「期間の定めなし」として常に対象にする。

BUG_IGNORE_VALIDITY=1 にすると、読み込んではいるが検索では使わない状態になる
（本番の現状。payload に入っているのに、旧版も新版も同じように出てくる）。
"""
from __future__ import annotations

import datetime as dt
import re

from . import config

_DATE = r"(\d{4})-(\d{2})-(\d{2})"
_RANGE_RE = re.compile(rf"適用期間[：:]\s*{_DATE}\s*[〜~～-]\s*{_DATE}")
_FROM_RE = re.compile(rf"適用開始[：:]\s*{_DATE}")
_UNTIL_RE = re.compile(rf"適用終了[：:]\s*{_DATE}")


def _num(y: str, m: str, d: str) -> int:
    """2026-04-01 → 20260401。Qdrant の Range で大小比較できるよう整数にする。"""
    return int(y + m + d)


def parse(text: str) -> tuple[int | None, int | None]:
    """本文から (valid_from, valid_until) を読み取る。無ければ None。"""
    if m := _RANGE_RE.search(text):
        return _num(*m.groups()[:3]), _num(*m.groups()[3:])
    f = _FROM_RE.search(text)
    u = _UNTIL_RE.search(text)
    return (_num(*f.groups()) if f else None, _num(*u.groups()) if u else None)


def as_of() -> int:
    """基準日。AS_OF=2026-03-01 のように指定すると、その日の時点で検索する。"""
    d = dt.date.fromisoformat(config.AS_OF) if config.AS_OF else dt.date.today()
    return int(d.strftime("%Y%m%d"))


def is_valid(valid_from: int | None, valid_until: int | None, day: int | None = None) -> bool:
    day = day or as_of()
    return (valid_from is None or valid_from <= day) and (valid_until is None or day <= valid_until)


def fmt(valid_from: int | None, valid_until: int | None) -> str:
    """表示用。期間の定めがなければ空文字。"""
    if valid_from is None and valid_until is None:
        return ""
    s = lambda n: f"{str(n)[:4]}-{str(n)[4:6]}-{str(n)[6:]}" if n else ""  # noqa: E731
    return f"{s(valid_from)}〜{s(valid_until)}"
