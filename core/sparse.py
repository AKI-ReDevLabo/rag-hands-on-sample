"""スパースベクトル（BM25 相当）の生成。

第5章「事件簿①：sparseベクトルが効かない」の中心にあるコード。

【壊れた実装】 abs(hash(token)) % DIM
  Python の文字列 hash は PYTHONHASHSEED によりプロセス起動ごとに
  ランダム化される。したがって
      登録時のプロセス と 検索時のプロセス
  でトークンが別のインデックスに落ちる。
  同一プロセス内では一貫しているため、単体テストは通ってしまう。

【正しい実装】 blake2b(token) % DIM
  プロセスをまたいでも同じトークンは同じインデックスに落ちる。

BUG_SPARSE_UNSTABLE_HASH=1 で壊れた実装に切り替わる。
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from . import config

# --- トークナイザ ---------------------------------------------------------
# SudachiPy があれば形態素解析、なければ文字 2-gram / 3-gram にフォールバックする。
try:
    from sudachipy import dictionary, tokenizer as _sudachi_tokenizer

    _tokenizer_obj = dictionary.Dictionary(dict="core").create()
    _split_mode = _sudachi_tokenizer.Tokenizer.SplitMode.C
    HAS_SUDACHI = True
except Exception:  # noqa: BLE001 - 辞書未インストールなども含めて握る
    _tokenizer_obj = None
    _split_mode = None
    HAS_SUDACHI = False


def tokenize(text: str) -> list[str]:
    """テキストをトークン列にする。

    1文字トークンを一律で捨てると「秀」「優」「函」のような
    日本語で意味のある1文字語が引けなくなる。
    ひらがな1文字だけを落とす。
    """
    if HAS_SUDACHI:
        raw = [m.surface() for m in _tokenizer_obj.tokenize(text, _split_mode)]
    else:
        raw = _char_ngrams(text)

    tokens = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        if len(t) == 1 and _is_hiragana(t):
            continue
        tokens.append(t)
    return tokens


def _is_hiragana(ch: str) -> bool:
    return "\u3041" <= ch <= "\u309f"


def _char_ngrams(text: str) -> list[str]:
    """SudachiPy が無い環境向けのフォールバック。文字 2-gram と 3-gram。"""
    text = "".join(text.split())
    grams: list[str] = []
    for n in (2, 3):
        grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return grams


# --- インデックス割り当て -------------------------------------------------
def token_to_index(token: str) -> int:
    """トークンをスパースベクトルのインデックスに写す。"""
    if config.BUG_SPARSE_UNSTABLE_HASH:
        # 壊れた実装。プロセスごとに結果が変わる。
        return abs(hash(token)) % config.SPARSE_DIM
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % config.SPARSE_DIM


def make_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """(indices, values) を返す。

    異なるトークンが同じインデックスに衝突しうるので、
    defaultdict で値を合算してから返す。
    ここを怠ると Qdrant が indices の重複を拒否して 422 を返す。
    """
    tokens = tokenize(text)
    if not tokens:
        return [], []

    counts = Counter(tokens)
    total = sum(counts.values())

    merged: dict[int, float] = defaultdict(float)
    for token, cnt in counts.items():
        idx = token_to_index(token)
        tf = cnt / total
        # 長いトークンほど情報量が多いとみなす簡易な重み
        weight = 1.0 if len(token) >= 3 else 0.7
        merged[idx] += tf * weight

    indices = list(merged.keys())
    values = [round(v, 6) for v in merged.values()]
    return indices, values
