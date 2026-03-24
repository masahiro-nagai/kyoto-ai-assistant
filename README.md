# 京都 Local Explorer

地元民目線の京都AIアシスタント。天候・混雑・季節に合わせたスポットを提案します。

## 概要

Vertex AI（Gemini 2.5 Flash）と Google Search Grounding を活用し、旅行者・地元民に「地元民目線の京都体験」を提供するWebアプリです。

## 技術スタック

| 項目 | 技術 |
|------|------|
| フロントエンド | Streamlit |
| AIモデル | Vertex AI Gemini 2.5 Flash |
| 検索グラウンディング | Google Search |
| ホスティング | Cloud Run（asia-northeast1）※Phase 3以降 |
| データストア | Firestore（favorites / wishlist）※Phase 4以降 |
| RAG | Cloud Storage + Vertex AI Vector Search ※Phase 2以降 |

## フェーズ進捗

| Phase | 内容 | 状態 |
|-------|------|------|
| Phase 1 | 基本チャット + Google Search Grounding | ✅ 完了 |
| Phase 2 | RAGデータ注入（隠れ家スポット） | 🔜 |
| Phase 3 | Cloud Run デプロイ / React UI移行 | 🔜 |
| Phase 4 | Firebase Auth + favorites保存 | 🔜 |
| Phase 5 | wishlist + 類似推薦 | 🔜 |

## セットアップ

### 前提

- Python 3.12+
- GCPプロジェクト（Vertex AI API 有効化済み）
- [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc) 設定済み

```bash
gcloud auth application-default login
```

### インストール

```bash
git clone https://github.com/masahiro-nagai/kyoto-ai-assistant.git
cd kyoto-ai-assistant
pip install -r requirements.txt
```

### 環境変数

```bash
cp .env.example .env
# .envを編集してプロジェクトIDを設定
```

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `GOOGLE_CLOUD_PROJECT` | GCPプロジェクトID | `kyoto-ai-assistant` |
| `GOOGLE_CLOUD_LOCATION` | リージョン | `asia-northeast1` |

### ローカル起動

```bash
GOOGLE_CLOUD_PROJECT=kyoto-ai-assistant \
GOOGLE_CLOUD_LOCATION=asia-northeast1 \
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

## ファイル構成

```
kyoto-ai-assistant/
├── app.py                  # Streamlitメインアプリ
├── prompts/
│   └── system_prompt.txt   # AIキャラクター・出力フォーマット定義
├── firestore.rules         # Firestore Security Rules
├── firebase.json           # Firebase設定
├── requirements.txt        # Pythonパッケージ
├── Dockerfile              # Cloud Run向けコンテナ
├── .env.example            # 環境変数サンプル
└── SPEC.md                 # 仕様書（Source of Truth）
```

## 使い方

1. テキストボックスに質問を入力（例：「今日雨が降ってるんだけどどこ行こう？」）
2. 現在地を任意で入力（例：「祇園」）
3. 「提案してもらう」をクリック
4. スポットカード（名前・推薦理由・地図リンク）が表示される

## 注意事項

- 営業時間・混雑状況・イベント情報は変動します。お出かけ前に最新情報をご確認ください。
- Vertex AI の利用にはGCPの課金が発生します。月額$100以内を目安に運用しています。

## ライセンス

MIT
