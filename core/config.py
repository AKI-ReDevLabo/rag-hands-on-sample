"""設定の一元管理。

環境変数は .env から読む。値の意味は README を参照。
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # このファイルは <repo>/core/config.py なので parents[1] がリポジトリルート
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv が無くても環境変数だけで動く
    pass


# --- Qdrant ---------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "minefield_docs")

# --- 埋め込み -------------------------------------------------------------
# local  : sentence-transformers（APIキー不要・完全ローカル）
# gemini : Vertex AI text-multilingual-embedding-002
# dummy  : ハッシュベースの決定的ダミー（モデル未DLでもパイプラインを試せる）
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "intfloat/multilingual-e5-small")
LOCAL_VECTOR_SIZE = int(os.getenv("LOCAL_VECTOR_SIZE", "384"))
# local の計算に使うデバイス。auto / cpu / cuda など。
# auto は「インストールされた PyTorch が対応している GPU のときだけ」cuda を使い、
# それ以外（古い GPU・GPU なし）は cpu にする。
LOCAL_DEVICE = os.getenv("LOCAL_DEVICE", "auto")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "text-multilingual-embedding-002")
GEMINI_VECTOR_SIZE = int(os.getenv("GEMINI_VECTOR_SIZE", "768"))
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "asia-northeast1")

DUMMY_VECTOR_SIZE = int(os.getenv("DUMMY_VECTOR_SIZE", "256"))

# --- チャンク -------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "300"))

# --- 検索 -----------------------------------------------------------------
SPARSE_DIM = int(os.getenv("SPARSE_DIM", "50000"))
N_RESULTS = int(os.getenv("N_RESULTS", "15"))
RRF_SCORE_THRESHOLD = float(os.getenv("RRF_SCORE_THRESHOLD", "0.20"))

# --- 地雷スイッチ ---------------------------------------------------------
# True にすると sparse ベクトルの生成を「壊れた実装」に切り替える。
# 第5章の事件簿を読者が手元で再現するためのフラグ。
BUG_SPARSE_UNSTABLE_HASH = os.getenv("BUG_SPARSE_UNSTABLE_HASH", "0") == "1"
# True にすると差分登録を本番の旧実装に戻す。登録済みのファイルには
# 「足りない chunk_index を足すだけ」で、古い index も古い内容も消さない（孤児チャンクの再現）。
BUG_ORPHAN_CHUNKS = os.getenv("BUG_ORPHAN_CHUNKS", "0") == "1"
# 短いチャンクの併合で、CHUNK_SIZE の上限を見ない（上限チェックを足す前の実装）。
# 見出しや表が隣の段落と結合され、CHUNK_SIZE を超えるチャンクができる。
BUG_MERGE_NO_LIMIT = os.getenv("BUG_MERGE_NO_LIMIT", "0") == "1"
# 管理番号を日付の一部からも拾う旧い正規表現に戻す（2026-09-14 から 2026-09 を拾う）。
BUG_CODE_REGEX_MATCHES_DATE = os.getenv("BUG_CODE_REGEX_MATCHES_DATE", "0") == "1"
# 適用期間を読み込むが、検索の絞り込みには使わない（本番の現状。旧版の規格も出てくる）。
BUG_IGNORE_VALIDITY = os.getenv("BUG_IGNORE_VALIDITY", "0") == "1"
# 事件簿③：検索不要と判定された追加質問を、フリーモード用プロンプトに回す（人格が入れ替わる）。
BUG_FREE_MODE_ROUTING = os.getenv("BUG_FREE_MODE_ROUTING", "0") == "1"
# 事件簿③：共通ルールに「前の回答には言及しない」を足す（continuation の指示と矛盾する）。
BUG_CONTRADICTING_RULE = os.getenv("BUG_CONTRADICTING_RULE", "0") == "1"

# --- LLM ------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-4-6")
# 小さくすると回答が途中で切れる（stop_reason=max_tokens の再現）。
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
# 追加質問のときに LLM へ渡す直前のやりとりの数（質問と回答の組の数）。
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))

# --- リランク -------------------------------------------------------------
# APIキーがあるとき、検索で広めに取った候補を Haiku に採点させて並べ替える。
# 「わからない」の判定も、RRF の閾値ではなく Haiku の最高スコアで行う。
# 0 にすると RRF の順位と閾値だけで絞る（APIキーが無いときも同じ）。
RERANK = os.getenv("RERANK", "1") == "1"
# haiku   : Haiku に採点させる（APIキーが必要。本番と同じ）
# keyword : 質問の語が本文にいくつ含まれるかで採点する（APIキー不要の簡易版）
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "haiku")
RERANK_MODEL = os.getenv("RERANK_MODEL", "claude-haiku-4-5")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "40"))  # 広めに取る候補数
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))  # 回答生成に渡す件数
# Haiku の最高スコア（0〜10）がこれ未満なら「わからない」（nohit）とする。
RERANK_NO_HIT_SCORE = int(os.getenv("RERANK_NO_HIT_SCORE", "3"))

# --- コード指定モード -----------------------------------------------------
# 質問に管理番号（KR-3104-01 など）があれば、payload の codes で絞り込んで検索する。
# 0 にすると、コードも普通の言葉として検索する（dense でも sparse でも引けない状態）。
CODE_SEARCH = os.getenv("CODE_SEARCH", "1") == "1"

# --- 適用期間 -------------------------------------------------------------
# 検索の基準日（YYYY-MM-DD）。空なら今日。旧版が有効だった日を指定して確かめる用。
AS_OF = os.getenv("AS_OF", "")

# --- 追加質問の書き直し ---------------------------------------------------
# 「それの締切は？」のような省略された追加質問を、直前のやりとりを元に
# 検索用の一文に書き直してから検索する（APIキーがあるときだけ）。
EXPAND_QUERY = os.getenv("EXPAND_QUERY", "1") == "1"
EXPAND_MODEL = os.getenv("EXPAND_MODEL", "claude-haiku-4-5")
# 書き直しに使う直前のメッセージ数（質問と回答を別々に数える）。
EXPAND_HISTORY_MESSAGES = int(os.getenv("EXPAND_HISTORY_MESSAGES", "4"))

# --- 質問ログ -------------------------------------------------------------
# チャットUIの質問を JSON Lines で追記する。空にすると記録しない。
QUESTION_LOG = os.getenv(
    "QUESTION_LOG", str(Path(__file__).resolve().parents[1] / "logs" / "questions.jsonl")
)

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
# data/ に加えて登録するフォルダ。事件簿①のノイズ文書（data_noise/）を足すときに使う。
# 本番用のコレクションに混ぜないよう、COLLECTION_NAME も別名にすること。
EXTRA_DATA_DIR = os.getenv("EXTRA_DATA_DIR", "")

# 構造化データ層（structured/）。「何件」「いくつ」のような質問を、RAG ではなく表から答える。
# 0 にすると、すべての質問を RAG で答える（事件簿⑤の「答えられない」状態）。
STRUCTURED = os.getenv("STRUCTURED", "1") == "1"

# 登録台帳（SQLite）。どのファイルを何チャンク登録したかを記録する。
LEDGER_PATH = os.getenv(
    "LEDGER_PATH", str(Path(__file__).resolve().parents[1] / "state" / "ledger.sqlite3")
)


def vector_size() -> int:
    """現在の埋め込みバックエンドに対応する次元数を返す。"""
    if EMBEDDING_BACKEND == "gemini":
        return GEMINI_VECTOR_SIZE
    if EMBEDDING_BACKEND == "dummy":
        return DUMMY_VECTOR_SIZE
    return LOCAL_VECTOR_SIZE
