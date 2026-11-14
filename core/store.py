"""Qdrant への登録・削除・検索。

dense と sparse を名前付きベクトルとして持つコレクションを作り、
Prefetch + RRF でハイブリッド検索する。

【注意】Qdrant のベクトル定義（dense / sparse の有無や次元）は
コレクション作成時に確定し、あとから変更できない。
payload はスキーマレスなので自由に足せる。
この非対称性が事故のもとになる。
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from . import config, embedder, sparse
from . import validity
from .codes import extract_codes

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        # timeout はデフォルトのままだと、WSL2からDocker Desktop経由で
        # 接続する環境で create_payload_index 等がまれにタイムアウトする。
        _client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    return _client


def collection_exists() -> bool:
    return client().collection_exists(config.COLLECTION_NAME)


def ensure_collection(recreate: bool = False) -> None:
    """コレクションを用意する。

    既存コレクションがある場合、定義が古いままでも「存在します」で
    素通りしてしまうのが典型的な事故。sparse 定義の有無を必ず検査する。
    """
    c = client()
    if recreate and collection_exists():
        c.delete_collection(config.COLLECTION_NAME)

    if not collection_exists():
        c.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=config.vector_size(),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        for field in ("source", "category", "codes"):
            c.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        for field in ("chunk_index", "valid_from", "valid_until"):
            c.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        return

    info = c.get_collection(config.COLLECTION_NAME)
    if not info.config.params.sparse_vectors:
        raise RuntimeError(
            "既存コレクションに sparse 定義がありません。"
            " ingest.py --recreate で作り直してください。"
        )


def _source_filter(source: str, *extra: models.Condition) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value=source)), *extra]
    )


def delete_by_source(source: str) -> int:
    """指定ファイル由来のチャンクをすべて削除する。"""
    client().delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=_source_filter(source)),
    )
    return 1


def delete_indices_from(source: str, start: int) -> None:
    """指定ファイルの chunk_index が start 以上のチャンクを削除する。

    分割の仕方が変わってチャンク数が減ったとき、もう存在しない番号を消すためのもの。
    """
    client().delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=_source_filter(
                source, models.FieldCondition(key="chunk_index", range=models.Range(gte=start))
            )
        ),
    )


def existing_indices(source: str) -> set[int]:
    """Qdrant に実際に入っている、指定ファイルの chunk_index の一覧。"""
    found: set[int] = set()
    offset = None
    while True:
        points, offset = client().scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=_source_filter(source),
            limit=256,
            offset=offset,
            with_payload=["chunk_index"],
            with_vectors=False,
        )
        found.update(p.payload.get("chunk_index") for p in points)
        if offset is None:
            return found


def list_sources() -> set[str]:
    """コレクションに実際に入っている source の一覧を返す。

    「何を登録したか」の記録ではなく、ベクトルDBの実体を数える。
    """
    sources: set[str] = set()
    offset = None
    while True:
        points, offset = client().scroll(
            collection_name=config.COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )
        sources.update(p.payload.get("source") for p in points)
        if offset is None:
            return sources


def delete_orphans(existing: set[str]) -> list[str]:
    """data/ に無くなった文書のチャンクを削除し、削除した source を返す。

    登録処理は「今あるファイル」しか見ないため、ファイルを消しても
    そのチャンクはコレクションに残り続ける。実体と突き合わせて消す。
    """
    orphans = sorted(list_sources() - existing)
    return [source for source in orphans if delete_by_source(source)]


def point_id(source: str, chunk_index: int) -> str:
    """ファイル名とチャンク番号から決まる ID。同じ番号を登録すると上書きになる。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-hands-on/{source}#{chunk_index}"))


def upsert_chunks(
    source: str,
    category: str,
    chunks: list[str],
    indices: list[int] | None = None,
    valid: tuple[int | None, int | None] = (None, None),
) -> int:
    """チャンク列を登録する。indices を渡すと、その番号のチャンクだけを登録する。

    valid は文書の適用期間 (valid_from, valid_until)。ファイル単位なので全チャンクに同じ値を入れる。

    ID はファイル名と番号から決まるので、同じ番号は上書きになる。
    古い番号の削除は呼び出し側（ingest.py）の責任。
    """
    targets = range(len(chunks)) if indices is None else sorted(indices)
    targets = [i for i in targets if 0 <= i < len(chunks)]
    if not targets:
        return 0

    dense_vectors = embedder.embed([chunks[i] for i in targets])
    points = []
    for i, dense in zip(targets, dense_vectors):
        indices_, values = sparse.make_sparse_vector(chunks[i])
        points.append(
            models.PointStruct(
                id=point_id(source, i),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(indices=indices_, values=values),
                },
                payload={
                    "source": source,
                    "category": category,
                    "chunk_index": i,
                    "text": chunks[i],
                    "codes": extract_codes(chunks[i]),  # 管理番号（コード指定モード用）
                    # 適用期間（YYYYMMDD の整数）。定めがなければキー自体を入れない
                    **({"valid_from": valid[0]} if valid[0] else {}),
                    **({"valid_until": valid[1]} if valid[1] else {}),
                },
            )
        )
    client().upsert(collection_name=config.COLLECTION_NAME, points=points)
    return len(points)


def count() -> int:
    return client().count(config.COLLECTION_NAME, exact=True).count


def _validity_conditions(day: int) -> list[models.Condition]:
    """基準日 day に有効なチャンクだけを残す条件。期間の定めが無い（キーが無い）ものは残す。"""
    return [
        models.Filter(
            should=[
                models.IsEmptyCondition(is_empty=models.PayloadField(key="valid_from")),
                models.FieldCondition(key="valid_from", range=models.Range(lte=day)),
            ]
        ),
        models.Filter(
            should=[
                models.IsEmptyCondition(is_empty=models.PayloadField(key="valid_until")),
                models.FieldCondition(key="valid_until", range=models.Range(gte=day)),
            ]
        ),
    ]


def search(
    query: str, mode: str = "hybrid", limit: int | None = None, codes: list[str] | None = None
) -> list[dict]:
    """検索する。

    mode:
      hybrid : dense + sparse を RRF で融合（既定）
      dense  : 密ベクトルのみ
      sparse : スパースのみ

    codes を渡すと、payload の codes にそのどれかを持つチャンクだけに絞る（コード指定モード）。
    基準日に適用期間外の文書は除く（BUG_IGNORE_VALIDITY=1 なら除かない）。
    hybrid では dense と sparse の両方の候補を、同じ条件で絞ってから混ぜる。

    RRF の融合スコアは 1/(k+rank) の和になる。
    両者に重複が1件も無いとスコアが単一分数（1/2, 1/3, ...）に揃う。
    全部が単一分数なら、片方が無関係な結果を返している疑いが濃い。
    """
    limit = limit or config.N_RESULTS
    c = client()
    must: list[models.Condition] = []
    if codes:
        must.append(models.FieldCondition(key="codes", match=models.MatchAny(any=codes)))
    if not config.BUG_IGNORE_VALIDITY:
        must.extend(_validity_conditions(validity.as_of()))
    flt = models.Filter(must=must) if must else None

    if mode == "dense":
        res = c.query_points(
            collection_name=config.COLLECTION_NAME,
            query=embedder.embed_one(query),
            using="dense",
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
    elif mode == "sparse":
        indices, values = sparse.make_sparse_vector(query)
        res = c.query_points(
            collection_name=config.COLLECTION_NAME,
            query=models.SparseVector(indices=indices, values=values),
            using="sparse",
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
    elif mode == "hybrid":
        indices, values = sparse.make_sparse_vector(query)
        res = c.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=embedder.embed_one(query),
                    using="dense",
                    filter=flt,
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    filter=flt,
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    else:
        raise ValueError(f"未知の mode: {mode}")

    return [
        {
            "score": p.score,
            "source": p.payload.get("source"),
            "category": p.payload.get("category"),
            "text": p.payload.get("text", ""),
            "codes": p.payload.get("codes", []),
            "valid_from": p.payload.get("valid_from"),
            "valid_until": p.payload.get("valid_until"),
        }
        for p in res.points
    ]


def search_filtered(query: str, mode: str = "hybrid") -> list[dict]:
    """閾値でふるいにかけた検索結果を返す。

    ヒット0件を「わからない」と明示できることが、RAG の信頼性の土台になる。
    """
    hits = search(query, mode=mode)
    if mode == "hybrid":
        return [h for h in hits if h["score"] >= config.RRF_SCORE_THRESHOLD]
    return hits
