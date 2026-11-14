#!/usr/bin/env python3
"""検索して結果を表示する。

  python search.py "切り花 出荷"
  python search.py "乾燥ハーブ 返品" --mode sparse
  python search.py "拠点間異動" --answer    # LLM で回答まで生成
  python search.py "切り花 出荷" --no-rerank  # Haiku のリランクを通さない

APIキーがあれば、hybrid の結果は Haiku でリランクしてから表示する。
"""
from __future__ import annotations

import argparse

from core import answer as answer_mod
from core import config, store, validity
from core.retrieve import retrieve
from structured.dispatcher import dispatch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    ap.add_argument("--raw", action="store_true", help="リランクも閾値もかけない")
    ap.add_argument("--no-rerank", action="store_true", help="リランクを通さない")
    ap.add_argument("--answer", action="store_true", help="LLM で回答を生成する")
    args = ap.parse_args()

    if args.raw:
        hits, r = store.search(args.query, mode=args.mode), None
    else:
        r = retrieve(args.query, mode=args.mode, use_rerank=not args.no_rerank)
        hits = r.hits

    structured = dispatch(args.query)

    print(f"query = {args.query!r}  mode={args.mode}  hits={len(hits)}")
    if r and r.reranked:
        print(
            f"（{'キーワード' if config.RERANK_BACKEND == 'keyword' else 'Haiku'} でリランク：最高スコア {r.top_score}"
            f"、{config.RERANK_NO_HIT_SCORE} 未満なら「わからない」）"
        )
    elif args.mode == "hybrid" and not args.raw:
        print(f"（RRF スコア閾値 {config.RRF_SCORE_THRESHOLD} で除外後）")
    if config.BUG_IGNORE_VALIDITY:
        print("!! BUG_IGNORE_VALIDITY=1 : 適用期間を読み込んでいますが、絞り込みに使いません")
    elif config.AS_OF:
        print(f"（基準日 {config.AS_OF} の時点で有効な文書だけを検索）")
    if r and r.note:
        print(f"!! {r.note}")
    print()

    for i, h in enumerate(hits, 1):
        head = h["text"].replace("\n", " ")[:70]
        rr = f"  rerank={h['rerank_score']:>2}" if h.get("rerank_score") is not None else ""
        period = validity.fmt(h.get("valid_from"), h.get("valid_until"))
        if period:
            expired = "" if validity.is_valid(h.get("valid_from"), h.get("valid_until")) else " 期間外!"
            period = f"  <{period}{expired}>"
        print(f"{i:>2}. {h['score']:.4f}{rr}  [{h['source']}]{period} {head}")

    if args.mode == "hybrid" and hits:
        scores = [h["score"] for h in hits]
        singles = sum(1 for s in scores if _looks_single_fraction(s))
        if singles >= len(scores) * 0.8:
            print()
            print("警告: スコアが単一分数に集中しています。")
            print("      dense と sparse の結果がまったく重なっていない可能性。")
            print("      sparse 側が無関係な結果を返していないか確認してください。")

    if structured:
        print("\n" + "-" * 60)
        print(f"構造化データで答えられる質問です（{structured.intent}）")
        print(f"  SQL : {structured.sql}")
        print(f"  回答: {structured.text}")

    if args.answer and not structured:
        print("\n" + "=" * 60)
        result = answer_mod.answer(args.query, hits)
        print(
            f"[state={result['state']} stop_reason={result['stop_reason']}"
            f" max_tokens={config.ANSWER_MAX_TOKENS}]"
        )
        print(result["text"])
        if result["truncated"]:
            print("\n!! 回答が max_tokens で打ち切られています")

    return 0


def _looks_single_fraction(score: float, k: int = 2, depth: int = 12) -> bool:
    """1/2, 1/3, 1/4 ... のような単一分数に見えるか。"""
    return any(abs(score - 1.0 / n) < 1e-6 for n in range(k, k + depth))


if __name__ == "__main__":
    raise SystemExit(main())
