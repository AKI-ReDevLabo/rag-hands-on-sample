"""検索して、回答生成に渡すチャンクを決める。

  追加質問なら、直前のやりとりを元に検索用の一文に書き直す（APIキーがあるとき）
    → hybrid 検索で広めに候補を取る
    → Haiku でリランク（APIキーがあり、RERANK=1 のとき。RERANK_BACKEND=keyword なら語の一致で）
    → 最高スコアが低ければ「わからない」（hits を空にする）

検索とリランクには、書き直した同じ文を使う。

リランクしないとき（APIキーが無い、RERANK=0、リランクが失敗した）は、
従来どおり RRF の閾値で絞る。ただし RRF の閾値では dense 側の上位が
必ず残るので、hits が空になる（nohit になる）ことはほぼない。
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, rerank, store
from .codes import extract_codes
from .expand import expand_query


@dataclass
class Retrieval:
    hits: list[dict]
    reranked: bool  # リランクが走ったか
    top_score: int | None = None  # リランクの最高スコア
    search_query: str | None = None  # 検索に使った文（書き直した場合だけ）
    note: str | None = None  # 書き直し・リランクをしなかった理由など、画面に出す一言


def _rrf_filter(hits: list[dict]) -> list[dict]:
    return [h for h in hits if h["score"] >= config.RRF_SCORE_THRESHOLD]


def _join(*notes: str | None) -> str | None:
    return " / ".join(n for n in notes if n) or None


def retrieve(
    query: str,
    mode: str = "hybrid",
    use_rerank: bool = True,
    history: list[dict] | None = None,
) -> Retrieval:
    ex = expand_query(query, history)
    q = ex.query
    sq = q if ex.expanded else None

    if mode != "hybrid":
        return Retrieval(hits=store.search(q, mode=mode), reranked=False, search_query=sq, note=ex.note)

    # コード指定モード：質問に管理番号があれば、codes で絞って検索する。
    # 絞った時点で候補は少ないので、リランクは掛けず RRF の閾値だけで判定する。
    codes = extract_codes(q) if config.CODE_SEARCH else []
    if codes:
        hits = _rrf_filter(store.search(q, mode=mode, codes=codes))
        return Retrieval(
            hits=hits,
            reranked=False,
            search_query=sq,
            note=_join(ex.note, f"コード指定モード：{'、'.join(codes)}"),
        )

    if not (use_rerank and config.RERANK and rerank.available()):
        return Retrieval(
            hits=store.search_filtered(q, mode=mode), reranked=False, search_query=sq, note=ex.note
        )

    candidates = store.search(q, mode=mode, limit=config.RERANK_CANDIDATES)
    result = rerank.rerank(q, candidates)

    if not result.used_rerank:
        # 失敗したときは、RRF の上位 N_RESULTS 件を閾値で絞る（従来の判定）。
        rr = f"リランクなし：{result.fallback_reason}" if result.fallback_reason else None
        return Retrieval(
            hits=_rrf_filter(candidates[: config.N_RESULTS]),
            reranked=False,
            search_query=sq,
            note=_join(ex.note, rr),
        )

    hits = result.hits if result.top_score >= config.RERANK_NO_HIT_SCORE else []
    return Retrieval(
        hits=hits, reranked=True, top_score=result.top_score, search_query=sq, note=ex.note
    )
