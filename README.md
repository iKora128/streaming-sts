# リアルタイム音声会話アプリケーション

このアプリケーションは、Google Cloud Speech-to-Text APIを使用してリアルタイムの音声認識を行い、認識されたテキストをGemini AIモデルに送信して応答を生成するStreamlitアプリケーションです。

## 機能

- マイクからのリアルタイム音声認識
- 短い発言には相槌、長い発言には会話を返す機能
- 会話履歴の表示と管理

## 必要条件

- Python 3.8以上
- Google Cloudアカウントと設定済みのプロジェクト
- Google Cloud Speech-to-Text APIの有効化
- Gemini APIキー

## インストール

1. リポジトリをクローンまたはダウンロードします。

2. 必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

3. `.env`ファイルを作成し、必要な環境変数を設定します。

```
# Google Cloud設定
GOOGLE_CLOUD_PROJECT=your-project-id

# API Keys
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key  # オプション
ANTHROPIC_API_KEY=your-anthropic-api-key  # オプション

# LLMモデル設定
ASSISTANT_MODEL=gemini-2.0-flash
HUMAN_MODEL=gpt-4o  # オプション
```

## 使い方

1. アプリケーションを起動します。

```bash
streamlit run app.py
```

2. ブラウザで表示されるStreamlitアプリケーションにアクセスします（通常は http://localhost:8501）。

3. 「録音開始」ボタンをクリックして音声認識を開始します。

4. マイクに向かって話しかけてください。

5. 短い発言（10文字未満）には相槌が、長い発言には会話が返されます。

6. 「録音停止」ボタンをクリックして音声認識を停止します。

7. 「履歴クリア」ボタンをクリックして会話履歴をクリアします。

## ファイル構成

- `app.py`: Streamlitアプリケーションのメインファイル
- `speech_to_text.py`: Google Cloud Speech-to-Text APIを使用するモジュール
- `llm_manager.py`: LLMモデル（Gemini, OpenAI, Anthropic）を管理するモジュール
- `requirements.txt`: 必要なPythonパッケージのリスト
- `.env`: 環境変数設定ファイル

## 注意事項

- マイクへのアクセス許可が必要です。
- Google Cloud Speech-to-Text APIの使用には課金が発生する場合があります。
- 音声認識の精度は環境やマイクの品質によって異なります。
