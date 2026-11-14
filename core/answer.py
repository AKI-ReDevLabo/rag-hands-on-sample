"""検索結果から回答を生成する。

状態ごとに別のプロンプトを組み立てる。

  hit          : 検索がヒットした
  nohit        : ヒットしなかった
  continuation : 直前の回答を受けた追加質問
  free         : 検索不要と判定された質問（相槌など）。社内文書を使わない

第5章「事件簿③：根拠があるのに『情報がありません』と言う」の再現スイッチ：

  BUG_FREE_MODE_ROUTING=1
    追加質問が「検索不要」と判定されたとき、continuation ではなく
    フリーモード用のプロンプトに回す。会話の途中で人格が入れ替わり、
    直前に自分で答えた内容を「その情報はありません」と否定する。
  BUG_CONTRADICTING_RULE=1
    共通ルールに「前の回答には言及しない」を足す。continuation の
    「直前の回答を撤回・否定しない」と矛盾する。
"""
from __future__ import annotations

from . import config

_COMMON = """あなたは社内文書を参照して質問に答えるアシスタントです。
参考資料に書かれていることだけを根拠にしてください。
推測で補わないでください。"""

# 壊れた実装で共通ルールに混じっていた1行。continuation の指示と矛盾する。
_CONTRADICTING_RULE = "前の回答には言及しないでください。"

_HIT = """{common}

以下の参考資料が見つかりました。これらを根拠に回答してください。
参考資料が見つかっている以上、「情報がありません」とは言わないでください。
資料が質問に部分的にしか答えていない場合は、答えられる範囲を示したうえで
何が不足しているかを述べてください。

# 参考資料
{context}"""

_NOHIT = """{common}

今回の質問に関連する社内文書は見つかりませんでした。
その事実を率直に伝えてください。推測で答えないでください。
どのような言葉で聞き直せば見つかりそうかを一言添えてください。"""

_CONTINUATION = """{common}

これは直前のやりとりの続きです。
直前の回答で述べた内容を、根拠なく撤回・否定しないでください。
新しい参考資料があれば、それを踏まえて補足してください。

# 参考資料
{context}"""

# 検索不要と判定された質問用。社内文書を使わない別の人格になる。
_FREE = """あなたは気さくな雑談相手です。
社内の規程・数字・商品の情報は持っていません。
社内のことを聞かれたら「その情報はありません」と答えてください。"""

_NO_SEARCH_CONTEXT = "（今回は検索していません。直前のやりとりを踏まえて答えてください）"


def _common() -> str:
    if config.BUG_CONTRADICTING_RULE:
        return _COMMON + "\n" + _CONTRADICTING_RULE
    return _COMMON


def _routes_to_free(is_continuation: bool, rag_needed: bool) -> bool:
    """フリーモード用のプロンプトを使うか。

    直した実装：検索不要でも、会話の途中（continuation）なら continuation のまま。
    壊れた実装：検索不要と判定されたら、会話の途中でもフリーモードに回す。
    """
    if rag_needed:
        return False
    return not is_continuation or config.BUG_FREE_MODE_ROUTING


def build_prompt(
    question: str, hits: list[dict], is_continuation: bool = False, rag_needed: bool = True
) -> str:
    """状態を判定して system プロンプトを組み立てる。"""
    if _routes_to_free(is_continuation, rag_needed):
        return _FREE
    context = "\n\n---\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    if is_continuation:
        if not rag_needed:
            context = _NO_SEARCH_CONTEXT
        return _CONTINUATION.format(common=_common(), context=context or "（なし）")
    if hits:
        return _HIT.format(common=_common(), context=context)
    return _NOHIT.format(common=_common())


def state_of(hits: list[dict], is_continuation: bool, rag_needed: bool = True) -> str:
    if _routes_to_free(is_continuation, rag_needed):
        return "free"
    if is_continuation:
        return "continuation"
    return "hit" if hits else "nohit"


def build_messages(question: str, history: list[dict] | None = None) -> list[dict]:
    """直前のやりとり＋今回の質問を messages にする。

    continuation のプロンプトは「直前の回答を否定するな」と指示するので、
    その直前の回答そのものを LLM に見せておかなければ意味がない。
    """
    turns = list(history or [])[-config.HISTORY_TURNS * 2 :] if config.HISTORY_TURNS > 0 else []
    # messages は user から始める必要がある
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return [{"role": t["role"], "content": t["content"]} for t in turns] + [
        {"role": "user", "content": question}
    ]


def answer(
    question: str,
    hits: list[dict],
    is_continuation: bool = False,
    history: list[dict] | None = None,
    rag_needed: bool = True,
) -> dict:
    """Anthropic API を呼んで回答を生成する。

    stop_reason を必ず持ち帰る。ここを見ていないと
    「回答が途中で切れている」ことに気づけない。
    history は [{"role": "user"|"assistant", "content": ...}] の列。
    """
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY が設定されていません")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANSWER_MODEL,
        max_tokens=config.ANSWER_MAX_TOKENS,
        system=build_prompt(question, hits, is_continuation, rag_needed),
        messages=build_messages(question, history),
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {
        "text": text,
        "stop_reason": resp.stop_reason,
        "truncated": resp.stop_reason == "max_tokens",
        "state": state_of(hits, is_continuation, rag_needed),
    }
