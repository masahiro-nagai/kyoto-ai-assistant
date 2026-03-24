"""
京都 Local Explorer - Phase 2 メインアプリ
Streamlit + Vertex AI Gemini 2.5 Flash + Google Search Grounding + 隐れ家RAG
"""

import json
import os
import re
import streamlit as st
from google import genai
from google.genai import types

# ─── 設定 ─────────────────────────────────────────────────────────────────────

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "kyoto-ai-assistant")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
MODEL_ID = "gemini-2.5-flash"
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
RAG_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "spots")

# Vertex AI クライアント初期化
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# ─── ユーティリティ ───────────────────────────────────────────────────────────

@st.cache_resource
def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


@st.cache_resource
def load_rag_data() -> str:
    """隐れ家スポットのMarkdownデータを読み込んで文字列として返す"""
    texts = []
    if not os.path.isdir(RAG_DATA_DIR):
        return ""
    for fname in sorted(os.listdir(RAG_DATA_DIR)):
        if fname.endswith(".md"):
            fpath = os.path.join(RAG_DATA_DIR, fname)
            with open(fpath, encoding="utf-8") as f:
                texts.append(f.read())
    return "\n\n".join(texts)


@st.cache_resource
def get_generation_config() -> types.GenerateContentConfig:
    system_prompt = load_system_prompt()
    google_search_tool = types.Tool(google_search=types.GoogleSearch())
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[google_search_tool],
        temperature=0.7,
        max_output_tokens=8192,
    )


def build_user_message(query: str, location: str) -> str:
    """ユーザー入力+現在地+RAGデータをまとめたメッセージを生成する"""
    rag_data = load_rag_data()
    parts = []
    if rag_data:
        parts.append(
            "## 地元民の隐れ家スポットデータ（優先参照）\n"
            + rag_data
            + "\n\n---\n"
        )
    if location.strip():
        parts.append(f"現在地: {location.strip()}")
    parts.append(query)
    return "\n\n".join(parts)


def parse_response(raw_text: str) -> dict | None:
    """AIの応答テキストからJSONを抽出してパースする"""
    # コードブロックがあれば除去
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()

    # まず全体をそのままパース試行
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ブレース追跡で最初の { ... } を抽出する（Grounding引用が末尾に付く場合に対応）
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(cleaned[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None



def render_spot_card(spot: dict) -> None:
    """スポット1件をカード形式で表示する"""
    name = spot.get("name", "不明")
    reason = spot.get("reason", "")
    maps_url = spot.get("maps_url", "")
    category = spot.get("category", "")
    area = spot.get("area", "")
    indoor = spot.get("indoor")
    crowd_note = spot.get("crowd_note", "")
    budget_level = spot.get("budget_level", "")
    stay_minutes = spot.get("stay_minutes")

    budget_emoji = {"low": "💰", "medium": "💰💰", "high": "💰💰💰"}.get(budget_level, "")
    indoor_tag = ("🏠 屋内" if indoor else "🌿 屋外") if indoor is not None else ""

    with st.container(border=True):
        # ヘッダー行
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {name}")
            if category or area:
                tags = " / ".join(filter(None, [category, area]))
                st.caption(f"📍 {tags}")
        with col2:
            if maps_url:
                st.link_button("🗺️ 地図を開く", maps_url, use_container_width=True)

        # 推薦理由
        st.write(reason)

        # メタ情報
        meta_parts = []
        if indoor_tag:
            meta_parts.append(indoor_tag)
        if budget_emoji:
            meta_parts.append(f"予算: {budget_emoji}")
        if stay_minutes:
            meta_parts.append(f"⏱️ 約{stay_minutes}分")
        if crowd_note:
            meta_parts.append(f"👥 {crowd_note}")

        if meta_parts:
            st.caption("　｜　".join(meta_parts))


def call_ai(user_message: str) -> tuple:
    """Vertex AI にリクエストし、パース済み応答を返す"""
    config = get_generation_config()
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=user_message,
        config=config,
    )
    raw = response.text
    return parse_response(raw), raw



# ─── UI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="京都 Local Explorer",
        page_icon="⛩️",
        layout="centered",
    )

    # ヘッダー
    st.title("⛩️ 京都 Local Explorer")
    st.caption("地元民目線の京都案内。天候・混雑・季節に合わせたスポットをご提案します。")
    st.divider()

    # 入力フォーム
    with st.form("query_form", clear_on_submit=False):
        query = st.text_area(
            "気になっていることや行きたい場所を教えてください",
            placeholder="例: 今日雨が降ってるんだけど、どこ行こう？",
            height=100,
        )
        location = st.text_input(
            "現在地（任意）",
            placeholder="例: 祇園、河原町、嵐山駅",
        )
        submitted = st.form_submit_button("🔍 提案してもらう", use_container_width=True)

    # 応答表示
    if submitted:
        if not query.strip():
            st.warning("質問を入力してください。")
            return

        user_message = build_user_message(query, location)

        with st.spinner("地元民が考え中どすえ…"):
            try:
                result, raw_text = call_ai(user_message)
            except Exception as e:
                st.error(f"エラーが発生しました。もう一度お試しください。\n\n詳細: {e}")
                return

        if result is None:
            # フォールバック：生テキストを表示
            st.warning("構造化データの取得に失敗しました。生の回答を表示します。")
            st.write(raw_text)
            st.info("再試行するか、質問の表現を変えてみてください。")
            return

        # サマリー
        if summary := result.get("summary"):
            st.info(f"💬 {summary}")

        # スポットカード一覧
        spots = result.get("spots", [])
        if spots:
            st.subheader(f"おすすめスポット（{len(spots)}件）")
            for spot in spots:
                render_spot_card(spot)
        else:
            st.warning("条件に合うスポットが見つかりませんでした。条件を変えて試してみてください。")

        # 全体アドバイス
        if advice := result.get("advice"):
            st.success(f"💡 {advice}")

        st.caption("※ 営業時間・混雑状況は変動します。お出かけ前に最新情報をご確認ください。")


if __name__ == "__main__":
    main()
