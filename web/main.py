"""チャットUI用のFastAPIサーバー。

会社の本番システムと同じ「チャット形式」を、最小構成で再現する。
認証・セッション保存・承認フローは持たない。持っているのは
検索・回答生成・状態(hit/nohit/continuation)の可視化だけ。

起動:
    uvicorn web.main:app --reload --port 8080
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import answer as answer_mod
from core import config, rerank, store
from core.intent import is_rag_needed
from core.retrieve import Retrieval, retrieve
from structured.dispatcher import dispatch

app = FastAPI(title="rag-hands-on chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    is_continuation: bool = False
    # 直前までのやりとり。追加質問のときに LLM へ渡す。
    history: list[Turn] = []


class Hit(BaseModel):
    source: str
    category: str | None = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    state: str  # hit / nohit / continuation / free / structured
    hits: list[Hit]
    answer: str | None
    stop_reason: str | None = None
    truncated: bool = False
    note: str | None = None
    # 追加質問を書き直して検索したときの、検索に使った文
    search_query: str | None = None


def log_question(
    question: str,
    state: str,
    hits: list[dict],
    truncated: bool,
    search_query: str | None = None,
) -> None:
    """質問を1行1件の JSON で追記する（第6章「質問ログから穴を見つける」）。

    記録に失敗しても回答は返す。ログのために利用者を止めない。
    """
    if not config.QUESTION_LOG:
        return
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "state": state,
        "hits": len(hits),
        "sources": [h["source"] for h in hits[:3]],
        "truncated": truncated,
        "search_query": search_query,
    }
    try:
        path = Path(config.QUESTION_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    """今どんな設定・地雷スイッチで動いているかを返す。

    デモや検証のときに「今どのバグを踏める状態か」を
    画面上で確認できるようにするためのエンドポイント。
    """
    return {
        "embedding_backend": config.EMBEDDING_BACKEND,
        "collection": config.COLLECTION_NAME,
        "chunk_count": store.count() if store.collection_exists() else 0,
        "bug_sparse_unstable_hash": config.BUG_SPARSE_UNSTABLE_HASH,
        "bug_orphan_chunks": config.BUG_ORPHAN_CHUNKS,
        "bug_free_mode_routing": config.BUG_FREE_MODE_ROUTING,
        "bug_contradicting_rule": config.BUG_CONTRADICTING_RULE,
        "bug_ignore_validity": config.BUG_IGNORE_VALIDITY,
        "answer_max_tokens": config.ANSWER_MAX_TOKENS,
        "llm_available": bool(config.ANTHROPIC_API_KEY),
        "rrf_threshold": config.RRF_SCORE_THRESHOLD,
        "rerank": config.RERANK and rerank.available(),
        "rerank_backend": config.RERANK_BACKEND,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    history = [t.model_dump() for t in req.history] if req.is_continuation else []
    # 相槌のように検索の要らない質問は、検索しない（core/intent.py）。
    rag_needed = is_rag_needed(req.question)

    # 「何件」「いくつ」のような質問は、RAG ではなく構造化データで答える（第5章 事件簿⑤）。
    s = dispatch(req.question) if rag_needed else None
    if s:
        log_question(req.question, "structured", [], truncated=False)
        return ChatResponse(
            state="structured",
            hits=[],
            answer=s.text,
            note=f"構造化データで回答（{s.intent}）：{s.sql}",
        )

    if rag_needed:
        r = retrieve(req.question, mode="hybrid", history=history)
    else:
        r = Retrieval(hits=[], reranked=False, note="検索不要と判定（相槌）")
    hits = r.hits
    state = answer_mod.state_of(hits, req.is_continuation, rag_needed)

    hit_models = [
        Hit(
            source=h["source"],
            category=h.get("category"),
            score=round(h["score"], 4),
            snippet=h["text"].replace("\n", " ")[:120],
        )
        for h in hits
    ]

    # APIキー未設定でも検索結果までは見られるようにする。
    # 「LLMが使えないときにどう振る舞うか」自体が第7章の主題と重なる。
    if not config.ANTHROPIC_API_KEY:
        log_question(req.question, state, hits, truncated=False, search_query=r.search_query)
        return ChatResponse(
            state=state,
            hits=hit_models,
            answer=None,
            note="ANTHROPIC_API_KEY 未設定のため、検索結果のみ表示しています。",
            search_query=r.search_query,
        )

    result = answer_mod.answer(
        req.question,
        hits,
        req.is_continuation,
        history=history,
        rag_needed=rag_needed,
    )
    log_question(
        req.question, result["state"], hits, result["truncated"], search_query=r.search_query
    )
    return ChatResponse(
        state=result["state"],
        hits=hit_models,
        answer=result["text"],
        stop_reason=result["stop_reason"],
        truncated=result["truncated"],
        note=r.note,
        search_query=r.search_query,
    )
