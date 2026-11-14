"""検索で広めに取った候補を、Haiku に質問との関連度で採点させて並べ替える。

本番の listwise リランカーをサンプル向けに移したもの。

ハイブリッド検索の dense 側は、どんな質問にも「近い順」に何かを返す。
そのため RRF の閾値だけでは「関係のある文書が無い」を判定できない。
ここで Haiku が付けた最高スコアを見て、低ければ「わからない」（nohit）にする。

フェイルセーフ:
  API 呼び出しの失敗、max_tokens での打ち切り、JSON が読めない、のどれでも
  候補を RRF の順のまま返す（used_rerank=False）。呼び出し側は従来の
  RRF 閾値の判定に戻る。黙って戻らないよう、理由を fallback_reason に入れる。

RERANK_BACKEND=keyword にすると、Haiku の代わりに「質問の語が候補の本文に
いくつ含まれるか」で 0〜10 を付ける。APIキーが無くても、リランクが sparse の
ノイズを覆い隠す様子（事件簿①）や、nohit の判定を手元で再現するための簡易版。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from . import config, sparse

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    hits: list[dict]
    top_score: int | None  # Haiku の最高スコア（0〜10）。リランクしていなければ None
    used_rerank: bool  # Haiku のリランクが実際に走ったか
    fallback_reason: str | None = None  # 走らなかった理由（候補が少ない場合は None）
    scores: dict[int, int] = field(default_factory=dict)  # 候補の位置 → スコア


def build_rerank_prompt(query: str, candidates: list[dict]) -> str:
    listed = "\n\n".join(f"[{i}] {c['text'][:400]}" for i, c in enumerate(candidates))
    return f"""次の質問に対して、各文書がどれだけ直接的に役立つかを0〜10で採点してください。

質問: {query}

文書:
{listed}

採点ルール:
- 質問に直接答える情報を含む文書を高く、話題が近いだけ／無関係な文書は低く
- 品目名・拠点名・日付など固有名詞が一致するものを優先
- マークダウン記法は使わず、次のJSONのみを出力（前置き禁止）:
{{"scores": [{{"i": 0, "s": 8}}, {{"i": 1, "s": 3}}]}}"""


def parse_scores(raw: str) -> dict[int, int]:
    """Haiku の返答から {候補の位置: スコア} を取り出す。

    JSON をコードブロック（```）で囲んで返す癖があるので、先に取り除く。
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    return {int(item["i"]): int(item["s"]) for item in data["scores"]}


# 質問に必ず出てくるが、文書の中身とは関係のない語。
_QUERY_STOPWORDS = {"何", "どう", "どこ", "いつ", "誰", "教え", "ください", "ます", "です", "ですか", "について"}
_SYMBOL = re.compile(r"^[\W_ー]+$")


def query_terms(query: str) -> list[str]:
    """キーワード版リランクで照合する、質問の語。ひらがなだけの語（助詞など）と記号は除く。"""
    terms = []
    for t in sparse.tokenize(query):
        if t in _QUERY_STOPWORDS or _SYMBOL.match(t):
            continue
        if all(sparse._is_hiragana(ch) for ch in t):
            continue
        if t not in terms:
            terms.append(t)
    return terms


def keyword_scores(query: str, candidates: list[dict]) -> dict[int, int]:
    """質問の語のうち、本文に含まれる割合を 0〜10 にする（Haiku の採点の代わり）。"""
    terms = query_terms(query)
    if not terms:
        return {}
    return {
        i: round(10 * sum(1 for t in terms if t in c["text"]) / len(terms))
        for i, c in enumerate(candidates)
    }


def available() -> bool:
    """リランクを掛けられるか。keyword なら常に、haiku なら APIキーがあるとき。"""
    return config.RERANK_BACKEND == "keyword" or bool(config.ANTHROPIC_API_KEY)


def _fallback(candidates: list[dict], top_k: int, reason: str | None) -> RerankResult:
    if reason:
        logger.warning("リランクせずに RRF の順で返します: %s", reason)
    return RerankResult(
        hits=candidates[:top_k], top_score=None, used_rerank=False, fallback_reason=reason
    )


def rerank(query: str, candidates: list[dict], client=None, top_k: int | None = None) -> RerankResult:
    """候補（store.search の結果）を Haiku の採点順に並べ替える。

    返す hits には、元の dict に rerank_score を足したものが入る。
    """
    top_k = top_k or config.RERANK_TOP_K

    # 候補が少なければ並べ替える意味がない。RRF の順は呼び出し側で保証されている。
    if len(candidates) <= top_k:
        return _fallback(candidates, top_k, None)
    if config.RERANK_BACKEND == "keyword" and client is None:
        scores = keyword_scores(query, candidates)
        if not scores:
            return _fallback(candidates, top_k, "質問から照合する語を取り出せませんでした")
        return _ranked(candidates, scores, top_k)
    if not config.ANTHROPIC_API_KEY and client is None:
        return _fallback(candidates, top_k, "ANTHROPIC_API_KEY が設定されていません")

    import anthropic

    client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        resp = client.messages.create(
            model=config.RERANK_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": build_rerank_prompt(query, candidates)}],
        )
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
        return _fallback(candidates, top_k, f"Haiku の呼び出しに失敗しました（{type(e).__name__}）")

    # 採点が途中で切れると JSON が壊れる。事件簿④と同じ穴なので、先に見る。
    if resp.stop_reason == "max_tokens":
        return _fallback(candidates, top_k, "採点が max_tokens で途中で切れました")

    raw = "".join(b.text for b in resp.content if b.type == "text")
    try:
        scores = parse_scores(raw)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _fallback(candidates, top_k, "Haiku の採点結果を読めませんでした")

    # 範囲外の位置は捨てる。採点されなかった候補は最下位扱い。
    scores = {i: s for i, s in scores.items() if 0 <= i < len(candidates)}
    if not scores:
        return _fallback(candidates, top_k, "Haiku の採点結果が空でした")

    return _ranked(candidates, scores, top_k)


def _ranked(candidates: list[dict], scores: dict[int, int], top_k: int) -> RerankResult:
    # スコアの高い順。同点は元の順（RRF の順）を保つ。
    ranked = sorted(range(len(candidates)), key=lambda i: scores.get(i, -1), reverse=True)
    hits = [{**candidates[i], "rerank_score": scores.get(i)} for i in ranked[:top_k]]
    return RerankResult(hits=hits, top_score=max(scores.values()), used_rerank=True, scores=scores)
