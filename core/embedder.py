"""密ベクトル（dense）の生成。

バックエンドを3つ持つ。

  local  : sentence-transformers。APIキー不要・課金なし・完全ローカル。
           読者が何も契約せずに動かせるので既定値にしている。
  gemini : Vertex AI の text-multilingual-embedding-002。
           GCP プロジェクトと ADC が必要。
  dummy  : ハッシュベースの決定的ベクトル。モデルを落とさずに
           パイプライン全体の疎通を確認したいときに使う。検索品質は出ない。

バックエンドを差し替え可能にしておくと、あとから乗り換えるときに
呼び出し側を一切触らずに済む。
"""
from __future__ import annotations

import hashlib
import math

from . import config

_local_model = None
_gemini_model = None


def _resolve_local_device() -> str:
    """LOCAL_DEVICE=auto のとき、実際に動かせるデバイスを選ぶ。

    GPU が見えていても、インストールされた PyTorch がその GPU 世代
    （compute capability）向けのカーネルを含んでいなければ
    "no kernel image is available" で落ちる。そのときは cpu に逃がす。
    """
    if config.LOCAL_DEVICE != "auto":
        return config.LOCAL_DEVICE
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        return "cpu"
    return "cuda"


def _embed_local(texts: list[str]) -> list[list[float]]:
    global _local_model
    if _local_model is None:
        import warnings

        from sentence_transformers import SentenceTransformer

        with warnings.catch_warnings():
            # 非対応 GPU の警告は cpu に切り替えるので不要
            warnings.simplefilter("ignore", UserWarning)
            device = _resolve_local_device()
        _local_model = SentenceTransformer(config.LOCAL_MODEL_NAME, device=device)
    # e5 系は "query: " / "passage: " の接頭辞を推奨するが、
    # 本サンプルでは両者を同じ扱いにして単純さを優先する。
    vectors = _local_model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    global _gemini_model
    if _gemini_model is None:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        vertexai.init(project=config.GCP_PROJECT, location=config.GCP_LOCATION)
        _gemini_model = TextEmbeddingModel.from_pretrained(config.GEMINI_MODEL_NAME)
    # API のバッチ上限に配慮して小分けに送る
    out: list[list[float]] = []
    for i in range(0, len(texts), 5):
        batch = texts[i : i + 5]
        out.extend(e.values for e in _gemini_model.get_embeddings(batch))
    return out


def _embed_dummy(texts: list[str]) -> list[list[float]]:
    """文字 3-gram を決定的にハッシュして作る簡易ベクトル。

    意味は捉えられないが、同じ文字列には必ず同じベクトルが返る。
    パイプラインの疎通確認用。
    """
    size = config.DUMMY_VECTOR_SIZE
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * size
        s = "".join(text.split())
        for i in range(max(len(s) - 2, 1)):
            gram = s[i : i + 3]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % size] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def embed(texts: list[str]) -> list[list[float]]:
    """テキストのリストを密ベクトルのリストに変換する。"""
    if not texts:
        return []
    backend = config.EMBEDDING_BACKEND
    if backend == "gemini":
        return _embed_gemini(texts)
    if backend == "dummy":
        return _embed_dummy(texts)
    if backend == "local":
        return _embed_local(texts)
    raise ValueError(f"未知の EMBEDDING_BACKEND: {backend}")


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
