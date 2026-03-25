"""
京都 Local Explorer - Phase 4 メインアプリ
Streamlit + Vertex AI Gemini 2.5 Flash + Google Search Grounding + 隐れ家RAG + Firestore favorites
"""

import json
import os
import re
import uuid
import streamlit as st
from google import genai
from google.genai import types
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime, timezone

# ─── 設定 ─────────────────────────────────────────────────────────────────────

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "kyoto-ai-assistant")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
MODEL_ID = "gemini-2.5-flash"
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
RAG_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "spots")

# Vertex AI クライアント初期化
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# ─── Firebase Admin / Firestore ───────────────────────────────────────────────

@st.cache_resource
def init_firestore():
    """Firebase Admin SDKを初期化してFirestoreクライアントを返す"""
    if not firebase_admin._apps:
        # Cloud Run上はデフォルト認証（サービスアカウント）を使用
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
    return firestore.client()


def get_user_id() -> str:
    """セッション固有の匿名ユーザーIDを生成・維持する"""
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())
    return st.session_state["user_id"]


def save_favorite(spot: dict, user_id: str) -> bool:
    """スポットをFirestoreのfavoritesコレクションに保存し、session_stateキャッシュも更新する"""
    now = datetime.now(timezone.utc)
    fav_data = {
        "userId": user_id,
        "spot_name": spot.get("name", ""),
        "description": spot.get("reason", ""),
        "maps_url": spot.get("maps_url", ""),
        "category": spot.get("category", ""),
        "area": spot.get("area", ""),
        "saved_at": now,
    }
    # まずsession_stateキャッシュに追加（即時UI反映）
    if "favorites_cache" not in st.session_state:
        st.session_state.favorites_cache = []
    st.session_state.favorites_cache.insert(0, fav_data | {"id": f"local_{user_id}_{len(st.session_state.favorites_cache)}"})

    # Firestoreにも永続化（失敗してもUI反映は維持）
    try:
        db = init_firestore()
        _, doc_ref = db.collection("favorites").add(fav_data)
        # IDをFirestoreのものに更新
        st.session_state.favorites_cache[0]["id"] = doc_ref.id
        return True
    except Exception as e:
        st.warning(f"Firestore保存に失敗しました（ローカルには保存済み）: {e}")
        return True  # ローカル保存は成功しているのでTrueを返す


def load_favorites(user_id: str) -> list[dict]:
    """favorites一覧を取得する（session_stateキャッシュ優先、なければFirestoreから取得）"""
    # session_stateキャッシュがあればそちらを優先
    if "favorites_cache" in st.session_state:
        return st.session_state.favorites_cache

    # 初回のみFirestoreから取得してキャッシュに格納
    try:
        db = init_firestore()
        docs = (
            db.collection("favorites")
            .where(filter=FieldFilter("userId", "==", user_id))
            .stream()
        )
        results = [doc.to_dict() | {"id": doc.id} for doc in docs]
        results.sort(key=lambda x: x.get("saved_at") or 0, reverse=True)
        st.session_state.favorites_cache = results
        return results
    except Exception as e:
        st.error(f"お気に入りの読み込みに失敗しました: {e}")
        return []



def delete_favorite(doc_id: str) -> bool:
    """Firestoreからfavoritesドキュメントを削除する"""
    try:
        db = init_firestore()
        db.collection("favorites").document(doc_id).delete()
        return True
    except Exception as e:
        st.error(f"削除に失敗しました: {e}")
        return False


# ─── AI / RAG ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


@st.cache_resource
def load_rag_data() -> str:
    """隠れ家スポットのMarkdownデータを読み込んで文字列として返す"""
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
            "## 地元民の隠れ家スポットデータ（優先参照）\n"
            + rag_data
            + "\n\n---\n"
        )
    if location.strip():
        parts.append(f"現在地: {location.strip()}")
    parts.append(query)
    return "\n\n".join(parts)


def parse_response(raw_text: str) -> dict | None:
    """AIの応答テキストからJSONを抽出してパースする"""
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
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


# ─── UI コンポーネント ────────────────────────────────────────────────────────

def render_spot_card(spot: dict, user_id: str, show_save_btn: bool = True) -> None:
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
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"### {name}")
            if category or area:
                tags = " / ".join(filter(None, [category, area]))
                st.caption(f"📍 {tags}")
        with col2:
            if maps_url:
                st.link_button("🗺️ 地図", maps_url, use_container_width=True)
        with col3:
            if show_save_btn:
                btn_key = f"fav_{name}_{hash(reason)}"
                if st.button("⭐ 保存", key=btn_key, use_container_width=True):
                    if save_favorite(spot, user_id):
                        st.success("お気に入りに保存しました！")
                        st.rerun()

        st.write(reason)

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


def render_favorites_tab(user_id: str) -> None:
    """お気に入り一覧タブを表示する"""
    if st.button("🔄 更新", key="refresh_favorites"):
        st.rerun()

    favorites = load_favorites(user_id)

    if not favorites:
        st.info("まだお気に入りが保存されていません。スポット提案の「⭐ 保存」ボタンを押して追加してください。")
        return

    st.subheader(f"保存済みスポット（{len(favorites)}件）")
    for fav in favorites:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"### {fav.get('spot_name', '不明')}")
                if fav.get("category") or fav.get("area"):
                    tags = " / ".join(filter(None, [fav.get("category", ""), fav.get("area", "")]))
                    st.caption(f"📍 {tags}")
            with col2:
                maps_url = fav.get("maps_url", "")
                if maps_url:
                    st.link_button("🗺️ 地図", maps_url, use_container_width=True)
            with col3:
                if st.button("🗑️ 削除", key=f"del_{fav['id']}", use_container_width=True):
                    if delete_favorite(fav["id"]):
                        st.success("削除しました")
                        st.rerun()
            if fav.get("description"):
                st.write(fav["description"])
            saved_at = fav.get("saved_at")
            if saved_at:
                # Firestoreのタイムスタンプをdatetimeに変換
                if hasattr(saved_at, "seconds"):
                    saved_at = datetime.fromtimestamp(saved_at.seconds, tz=timezone.utc)
                st.caption(f"📅 保存日時: {saved_at.strftime('%Y/%m/%d %H:%M')}")


# ─── メイン UI ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="京都 Local Explorer",
        page_icon="⛩️",
        layout="centered",
    )

    user_id = get_user_id()

    st.title("⛩️ 京都 Local Explorer")
    st.caption("地元民目線の京都案内。天候・混雑・季節に合わせたスポットをご提案します。")
    st.divider()

    tab_suggest, tab_fav = st.tabs(["🔍 AI提案", "⭐ お気に入り"])

    # ─── タブ1: AI提案 ────────────────────────────────────────────────────────
    with tab_suggest:
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

        # フォーム送信時：AI呼び出し → session_stateに結果を保存
        if submitted:
            if not query.strip():
                st.warning("質問を入力してください。")
            else:
                user_message = build_user_message(query, location)
                with st.spinner("地元民が考え中どすえ…"):
                    try:
                        result, raw_text = call_ai(user_message)
                        st.session_state["last_result"] = result
                        st.session_state["last_raw_text"] = raw_text
                    except Exception as e:
                        st.error(f"エラーが発生しました。もう一度お試しください。\n\n詳細: {e}")
                        st.session_state.pop("last_result", None)

        # 結果の表示（session_stateから取得しボタン押下後のrerunでも維持）
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            raw_text = st.session_state.get("last_raw_text", "")

            if result is None:
                st.warning("構造化データの取得に失敗しました。生の回答を表示します。")
                st.write(raw_text)
                st.info("再試行するか、質問の表現を変えてみてください。")
            else:
                if summary := result.get("summary"):
                    st.info(f"💬 {summary}")

                spots = result.get("spots", [])
                if spots:
                    st.subheader(f"おすすめスポット（{len(spots)}件）")
                    for spot in spots:
                        render_spot_card(spot, user_id)
                else:
                    st.warning("条件に合うスポットが見つかりませんでした。条件を変えて試してみてください。")

                if advice := result.get("advice"):
                    st.success(f"💡 {advice}")

                st.caption("※ 営業時間・混雑状況は変動します。お出かけ前に最新情報をご確認ください。")

    # ─── タブ2: お気に入り ────────────────────────────────────────────────────
    with tab_fav:
        render_favorites_tab(user_id)



if __name__ == "__main__":
    main()
