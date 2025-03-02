"""
Speech-to-TextとLLMを連携させるStreamlitアプリケーション
"""
import os
import time
import threading
import queue
import logging
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
import traceback  # トレースバック情報を出力するために追加

from speech_to_text import SpeechToTextStreaming
from llm_manager import LLMManager

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# グローバル変数（スレッド間で共有）
_is_listening = False
_transcript_queue = queue.Queue()
_llm_manager = None  # LLMマネージャーをグローバル変数として保持
_transcripts = []  # 文字起こしを保存するグローバル変数
_responses = []  # 応答を保存するグローバル変数
_current_transcript = ""  # 現在の文字起こしを保存するグローバル変数
_current_response = ""  # 現在生成中のLLM応答を保存するグローバル変数
_update_ui = False  # UIの更新フラグ
_last_ui_update_time = time.time()  # 最後にUIを更新した時間
_force_update = False  # 強制更新フラグ
_is_generating = False  # LLM応答生成中フラグ
_accumulated_context = ""  # Global variable to accumulate short-turn transcripts

# ファイルベースの状態保存
_STATE_FILE = "app_state.json"

# セッション状態の初期化
if "stt" not in st.session_state:
    st.session_state.stt = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

if "transcripts" not in st.session_state:
    st.session_state.transcripts = []

if "responses" not in st.session_state:
    st.session_state.responses = []

if "current_transcript" not in st.session_state:
    st.session_state.current_transcript = ""

if "transcript_queue" not in st.session_state:
    st.session_state.transcript_queue = queue.Queue()

if "response_thread" not in st.session_state:
    st.session_state.response_thread = None

if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# セッション状態に判定結果を保存するための初期化
if 'turn_detection_results' not in st.session_state:
    st.session_state.turn_detection_results = []

# LLMのシステムプロンプト
AIZUCHI_SYSTEM_PROMPT = """
あなたは会話の相手です。ユーザーの発言に対して、相槌を打つように短く返答してください。
例えば「なるほど」「そうですね」「確かに」「それは興味深いですね」などの短い相槌を返してください。
返答は必ず1〜3語程度の短い相槌にしてください。長い説明や質問は避けてください。
"""

CONVERSATION_SYSTEM_PROMPT = """
あなたは会話の相手です。ユーザーの発言に対して、自然な会話を続けるように返答してください。
質問には答え、意見には共感や別の視点を提供し、会話を発展させてください。
返答は簡潔で自然な会話調にしてください。
"""

# 会話ターン判定用のシステムプロンプト
TURN_DETECTION_PROMPT = """
あなたは会話分析の専門家です。ユーザーの発言を分析し、それが完結した発言か、続きがある途中の発言かを判断してください。

以下の純粋なJSONのみを出力してください。他の説明は一切含めないでください：
{
  "continueConversation": true/false,
  "acknowledgement": "適切な短い相槌や返事"
}

判断基準：
- "continueConversation": false → 発言が完結している（質問や意見が明確に述べられている）
- "continueConversation": true → 発言が途中または続きがある（言いかけて止まっている、単語だけの発言など）

例：
- 「今日はどんな天気ですか？」→ {"continueConversation": false, "acknowledgement": "今日の天気についてお答えします"}
- 「今日は...」→ {"continueConversation": true, "acknowledgement": "はい"}
- 「それって」→ {"continueConversation": true, "acknowledgement": "はい？"}

会話が完結している場合は必ずfalseを返してください。特に質問や明確な意見表明があった場合は必ずfalseです。
"""

# グローバル変数の定義
_state_lock = threading.RLock()  # スレッドセーフな操作のためのロック

# 実験的な機能を有効化
st.set_page_config(
    page_title="リアルタイム音声会話",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "リアルタイム音声会話アプリケーション"
    }
)

# 更新間隔を短くする（実験的）
if "update_frequency" not in st.session_state:
    st.session_state.update_frequency = 0.1  # 100ミリ秒ごとに更新

def initialize_stt():
    """
    Speech-to-Textの初期化
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        error_msg = "環境変数GOOGLE_CLOUD_PROJECTが設定されていません。"
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    try:
        logger.info(f"Speech-to-Textを初期化します。プロジェクトID: {project_id}")
        stt = SpeechToTextStreaming(
            project_id=project_id,
            language_code="ja-JP",
            use_short_model=False
        )
        logger.info("Speech-to-Textの初期化に成功しました。")
        return stt
    except Exception as e:
        error_msg = f"Speech-to-Textの初期化中にエラーが発生しました: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None

def initialize_llm():
    """
    LLMの初期化
    """
    global _llm_manager
    try:
        logger.info("LLMを初期化します。")
        llm = LLMManager()
        _llm_manager = llm  # グローバル変数に保存
        logger.info("LLMの初期化に成功しました。")
        return llm
    except Exception as e:
        error_msg = f"LLMの初期化中にエラーが発生しました: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None

def on_speech_result(transcript: str, is_final: bool):
    """
    音声認識結果を受け取るコールバック関数
    
    Args:
        transcript: 認識されたテキスト
        is_final: 最終結果かどうか
    """
    global _transcript_queue, _current_transcript, _update_ui, _last_ui_update_time
    
    # 現在の文字起こしを更新
    _current_transcript = transcript
    
    # ログ出力
    if is_final:
        logger.info(f"音声認識結果（最終）: {transcript}")
    else:
        logger.debug(f"音声認識結果（中間）: {transcript}")
    
    # 最終結果の場合はキューに追加
    if is_final and transcript.strip():
        _transcript_queue.put(transcript)
        _update_ui = True  # UIの更新フラグをセット
        _last_ui_update_time = time.time()  # 最後の更新時間を記録

def on_llm_stream(chunk: str):
    """
    LLMからのストリーミング応答を処理するコールバック関数
    
    Args:
        chunk: LLMからのテキストチャンク
    """
    global _current_response, _update_ui, _last_ui_update_time, _force_update, _is_generating
    
    if chunk:
        _current_response += chunk
        _update_ui = True
        _last_ui_update_time = time.time()
        
        try:
            # このスレッドからセッション状態を更新
            st.session_state.current_response = _current_response
            # 強制更新要求
            _force_update = True
        except Exception as e:
            # セッション状態へのアクセスエラーは無視（別スレッドからのアクセスで発生する可能性あり）
            logger.warning(f"ストリーミングコールバックでセッション状態の更新中にエラー: {str(e)}")
            pass
    
    logger.debug(f"LLMストリーミング: {chunk}")  # 詳細なログ

def _save_state():
    """状態をファイルに保存"""
    state = {
        "transcripts": _transcripts,
        "responses": _responses,
        "current_transcript": _current_transcript,
        "current_response": _current_response,
        "turn_detection_results": st.session_state.get("turn_detection_results", [])  # 追加
    }
    
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        import json
        json.dump(state, f, ensure_ascii=False, indent=2)

def _load_state():
    """ファイルから状態を読み込む"""
    import json  # jsonモジュールをここでインポート
    
    if os.path.exists(_STATE_FILE):
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            # 既存の状態を読み込む
            _transcripts = state.get("transcripts", [])
            _responses = state.get("responses", [])
            _current_transcript = state.get("current_transcript", "")
            _current_response = state.get("current_response", "")
            
            # ターン判定結果の読み込み
            if "turn_detection_results" in state:
                st.session_state.turn_detection_results = state["turn_detection_results"]
            
            logger.info(f"ファイルから状態を読み込みました。転記数: {len(_transcripts)}, 応答数: {len(_responses)}")
            return _transcripts, _responses
        except Exception as e:
            logger.error(f"状態の読み込み中にエラーが発生しました: {str(e)}")
    
    return [], []

def process_transcripts():
    """
    音声認識結果を処理するスレッド関数
    """
    global _is_listening, _transcript_queue, _llm_manager, _transcripts, _responses, _update_ui, _last_ui_update_time, _force_update, _current_response, _is_generating, _accumulated_context
    
    logger.info("文字起こし処理スレッドを開始します。")
    import json
    import re
    
    while _is_listening:
        try:
            if not _transcript_queue.empty():
                # 文字起こしを取得した後
                transcript = _transcript_queue.get(timeout=0.1)
                logger.info(f"キューから取得した文字起こし: {transcript}")
                
                # グローバル変数に保存（これはスレッドセーフ）
                global _current_transcript
                _current_transcript = transcript
                
                # LLMを使用してターン判定
                turn_response = _llm_manager.call_model(
                    prompt=transcript,
                    system_prompt=TURN_DETECTION_PROMPT,
                    model="gemini-2.0-flash-lite",
                    stream=False
                )
                logger.info(f"ターン判定結果: {turn_response}")
                
                # 改善されたJSONパーサーを使用
                continue_conversation, ack = parse_turn_decision(turn_response, transcript)
                logger.info(f"解析結果: 会話継続={continue_conversation}, 相槌=\"{ack}\"")
                
                # 判定結果をセッション状態に保存
                turn_result = {
                    "transcript": transcript,
                    "continue_conversation": continue_conversation,
                    "acknowledgement": ack,
                    "raw_response": turn_response,
                    "timestamp": time.time()
                }
                
                # スレッドセーフに状態を更新
                with _state_lock:
                    # 最大10件まで保存
                    if "turn_detection_results" not in st.session_state:
                        st.session_state.turn_detection_results = []
                    
                    if len(st.session_state.turn_detection_results) >= 10:
                        st.session_state.turn_detection_results.pop(0)
                    st.session_state.turn_detection_results.append(turn_result)
                    _save_state()  # 状態を保存
                
                # 会話状態の更新と応答処理
                if continue_conversation:
                    # 会話継続の場合は相槌を返す
                    logger.info(f"会話継続と判断: 相槌=\"{ack}\"")
                    
                    # 相槌を表示するだけで、LLM応答は生成しない
                    with _state_lock:
                        _current_transcript = transcript
                        _current_response = ack
                        _update_ui = True
                        _last_ui_update_time = time.time()
                else:
                    # 会話完了の場合はLLM応答を生成
                    logger.info("会話完了と判断: 応答生成開始")
                    _is_generating = True
                    
                    try:
                        # LLM応答の生成
                        response_text = ""
                        
                        # 会話履歴を構築
                        conversation_history = ""
                        for i in range(min(len(_transcripts), len(_responses))):
                            conversation_history += f"ユーザー: {_transcripts[i]}\nAI: {_responses[i]}\n"
                        
                        # 現在の会話コンテキストを追加
                        current_context = f"{conversation_history}ユーザー: {transcript}\nAI: "
                        
                        # LLM応答の生成（モデル名を修正）
                        response_text = _llm_manager.call_model(
                            prompt=current_context,
                            system_prompt=CONVERSATION_SYSTEM_PROMPT,
                            model="gemini-2.0-flash-lite", 
                            stream=False
                        )
                        
                        logger.info(f"LLM応答生成完了: {response_text[:100]}...")
                        
                        # 応答を保存
                        with _state_lock:
                            _transcripts.append(transcript)
                            _responses.append(response_text)
                            _current_transcript = transcript
                            _current_response = response_text
                            _update_ui = True
                            _force_update = True
                            _last_ui_update_time = time.time()
                            _save_state()
                    except Exception as e:
                        logger.error(f"LLM応答生成中にエラーが発生しました: {str(e)}")
                        logger.error(str(e))
                        traceback.print_exc()
                    finally:
                        _is_generating = False
        except Exception as e:
            logger.error(f"文字起こしの処理中にエラーが発生しました: {str(e)}")
            logger.error(str(e))
            traceback.print_exc()
        
        time.sleep(0.1)  # CPUの使用率を下げるために短いスリープ

def parse_turn_decision(turn_response, transcript):
    """
    LLMを主体としたターン判定解析器（ヒューリスティックはバックアップのみ）
    """
    import json
    import re
    
    # デフォルト値
    continue_conversation = True
    ack = "なるほど"
    
    # 1. LLM応答からJSONを直接解析（メイン方法）
    try:
        json_match = re.search(r'\{.*\}', turn_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            turn_data = json.loads(json_str)
            continue_conversation = turn_data.get("continueConversation", True)
            ack = turn_data.get("acknowledgement", "なるほど")
            logger.info(f"LLM判定を使用: {continue_conversation}")
            return continue_conversation, ack
    except Exception as e:
        logger.warning(f"JSON解析エラー: {str(e)}")
    
    # 2. 正規表現で個別フィールド抽出（バックアップ1）
    try:
        continue_match = re.search(r'"continueConversation"\s*:\s*(true|false)', turn_response)
        ack_match = re.search(r'"acknowledgement"\s*:\s*"([^"]+)"', turn_response)
        
        if continue_match:
            continue_conversation = continue_match.group(1).lower() == "true"
            logger.info(f"continueConversation正規表現抽出: {continue_conversation}")
        
        if ack_match:
            ack = ack_match.group(1)
            logger.info(f"acknowledgement正規表現抽出: {ack}")
        
        if continue_match:  # continueConversationの値が抽出できていればOK
            return continue_conversation, ack
    except Exception as e:
        logger.warning(f"正規表現抽出エラー: {str(e)}")
    
    # 3. LLM応答テキスト内の単語に基づく簡易判定（バックアップ2）
    # これはまだLLMの応答に基づいている
    if "true" in turn_response.lower():
        continue_conversation = True
        logger.info("応答テキスト内の'true'に基づき会話継続と判定")
    elif "false" in turn_response.lower():
        continue_conversation = False
        logger.info("応答テキスト内の'false'に基づき会話完了と判定")
    
    # 最後に質問検出だけはオーバーライド（重要なケース）
    if "?" in transcript or "？" in transcript or any(q in transcript for q in ["何", "どう", "なぜ", "いつ", "どこ", "だれ", "誰", "ですか"]):
        continue_conversation = False
        logger.info("質問検出によるオーバーライド: 会話完了")
    
    return continue_conversation, ack

def start_listening():
    """
    音声認識を開始する
    """
    global _is_listening
    
    if st.session_state.is_listening:
        logger.info("すでに音声認識を開始しています。")
        return
    
    logger.info("音声認識を開始します。")
    
    # Speech-to-Textの初期化
    if not st.session_state.stt:
        logger.info("Speech-to-Textを初期化します。")
        st.session_state.stt = initialize_stt()
    
    # LLMの初期化
    if not st.session_state.llm:
        logger.info("LLMを初期化します。")
        st.session_state.llm = initialize_llm()
    
    if not st.session_state.stt or not st.session_state.llm:
        error_msg = "初期化に失敗しました。"
        logger.error(error_msg)
        st.error(error_msg)
        return
    
    # 音声認識を開始
    logger.info("マイクからの音声認識を開始します。")
    st.session_state.stt.start_listening(callback=on_speech_result)
    st.session_state.is_listening = True
    _is_listening = True
    
    # 文字起こし処理スレッドを開始
    logger.info("文字起こし処理スレッドを開始します。")
    st.session_state.response_thread = threading.Thread(target=process_transcripts)
    st.session_state.response_thread.daemon = True
    st.session_state.response_thread.start()

def stop_listening():
    """
    音声認識を停止する
    """
    global _is_listening
    
    if not st.session_state.is_listening:
        logger.info("音声認識はすでに停止しています。")
        return
    
    logger.info("音声認識を停止します。")
    
    # 音声認識を停止
    if st.session_state.stt:
        st.session_state.stt.stop_listening()
    
    st.session_state.is_listening = False
    _is_listening = False
    
    # スレッドが終了するのを待機
    if st.session_state.response_thread and st.session_state.response_thread.is_alive():
        logger.info("文字起こし処理スレッドの終了を待機します。")
        st.session_state.response_thread.join(timeout=1.0)
    
    logger.info("音声認識を停止しました。")

def clear_history():
    """
    会話履歴をクリアする
    """
    global _transcript_queue, _transcripts, _responses, _current_transcript, _update_ui, _accumulated_context
    
    logger.info("会話履歴をクリアします。")
    
    _transcripts = []
    _responses = []
    _current_transcript = ""
    _accumulated_context = ""  # 蓄積コンテキストもクリア
    
    # キューをクリア
    while not _transcript_queue.empty():
        _transcript_queue.get()
    
    # セッション状態も更新
    st.session_state.transcripts = []
    st.session_state.responses = []
    st.session_state.current_transcript = ""
    
    # ファイルに状態を保存
    _save_state()
    
    _update_ui = True
    
    logger.info("会話履歴をクリアしました。")

def update_session_state():
    """
    グローバル変数からセッション状態を更新する
    """
    global _transcripts, _responses, _current_transcript, _current_response, _force_update, _is_generating
    
    logger.info(f"セッション状態を更新します。転記数: {len(_transcripts)}, 応答数: {len(_responses)}")
    
    # 転記と応答の数が一致しない場合は調整
    if len(_transcripts) > len(_responses):
        logger.warning(f"転記数({len(_transcripts)})が応答数({len(_responses)})より多いです。調整します。")
        _transcripts = _transcripts[:len(_responses)]
    
    # ファイルから最新の状態を読み込む
    _load_state()
    
    # セッション状態を更新
    st.session_state.transcripts = _transcripts.copy()
    st.session_state.responses = _responses.copy()
    st.session_state.current_transcript = _current_transcript
    st.session_state.current_response = _current_response
    st.session_state.is_generating = _is_generating
    st.session_state.last_update_time = time.time()
    _force_update = False
    
    logger.info(f"セッション状態を更新しました。st.session_state.transcripts: {len(st.session_state.transcripts)}, st.session_state.responses: {len(st.session_state.responses)}")

def save_conversation_to_html():
    """
    会話履歴をHTML形式で保存する
    """
    global _transcripts, _responses
    
    if not _transcripts or not _responses:
        logger.info("保存する会話履歴がありません。")
        return None
    
    try:
        # HTMLテンプレート
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>会話履歴</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }
                .message {
                    display: flex;
                    margin-bottom: 10px;
                }
                .user-message {
                    justify-content: flex-end;
                }
                .ai-message {
                    justify-content: flex-start;
                }
                .message-bubble {
                    max-width: 70%;
                    padding: 10px 15px;
                    border-radius: 18px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                }
                .user-bubble {
                    background-color: #dcf8c6;
                    border-bottom-right-radius: 5px;
                }
                .ai-bubble {
                    background-color: #ffffff;
                    border-bottom-left-radius: 5px;
                }
                .timestamp {
                    font-size: 0.7em;
                    color: #999;
                    margin-top: 5px;
                    text-align: right;
                }
                h1 {
                    text-align: center;
                    color: #333;
                }
            </style>
        </head>
        <body>
            <h1>会話履歴</h1>
            <div class="chat-container">
                {chat_content}
            </div>
        </body>
        </html>
        """
        
        # 会話内容を生成
        chat_content = ""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        for i in range(len(_transcripts)):
            # HTMLエスケープ処理
            user_text = _transcripts[i].replace("<", "&lt;").replace(">", "&gt;")
            
            # ユーザーメッセージ
            chat_content += f"""
            <div class="message user-message">
                <div class="message-bubble user-bubble">
                    <div>{user_text}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
            """
            
            # AIメッセージ
            if i < len(_responses):
                # HTMLエスケープ処理
                ai_text = _responses[i].replace("<", "&lt;").replace(">", "&gt;")
                
                chat_content += f"""
                <div class="message ai-message">
                    <div class="message-bubble ai-bubble">
                        <div>{ai_text}</div>
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
                """
        
        # HTMLを生成
        html_content = html_template.format(chat_content=chat_content)
        
        # ファイル名を生成
        filename = f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.html"
        
        # HTMLファイルを保存
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"会話履歴をHTMLファイルに保存しました: {filename}")
        return filename
    except Exception as e:
        error_msg = f"会話履歴のHTML保存中にエラーが発生しました: {str(e)}"
        logger.error(error_msg)
        return None

def main():
    """
    メイン関数
    """
    # セッション状態の初期化
    if "transcripts" not in st.session_state:
        st.session_state.transcripts = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "current_transcript" not in st.session_state:
        st.session_state.current_transcript = ""
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""
    if "is_listening" not in st.session_state:
        st.session_state.is_listening = False
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = time.time()
    # ここに追加 - ターン判定結果の初期化
    if "turn_detection_results" not in st.session_state:
        st.session_state.turn_detection_results = []
    
    global _transcripts, _responses, _current_transcript, _current_response, _update_ui, _last_ui_update_time, _force_update, _is_generating
    
    logger.info("アプリケーションを開始します。")
    
    st.title("🎤 リアルタイム音声会話")
    
    # ファイルから状態を読み込む
    _load_state()
    
    # 起動時にグローバル変数からセッション状態を更新
    update_session_state()
    logger.info(f"起動時にセッション状態を更新しました。転記数: {len(st.session_state.transcripts)}, 応答数: {len(st.session_state.responses)}")
    
    # サイドバー
    with st.sidebar:
        st.header("設定")
        
        # 音声認識の開始/停止ボタン
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 録音開始", use_container_width=True, disabled=st.session_state.is_listening):
                logger.info("録音開始ボタンがクリックされました。")
                start_listening()
                st.rerun()  # UIを即時更新
        
        with col2:
            if st.button("⏹️ 録音停止", use_container_width=True, disabled=not st.session_state.is_listening):
                logger.info("録音停止ボタンがクリックされました。")
                stop_listening()
                st.rerun()  # UIを即時更新
        
        # 履歴クリアボタン
        if st.button("🗑️ 履歴クリア", use_container_width=True):
            logger.info("履歴クリアボタンがクリックされました。")
            clear_history()
            st.rerun()  # UIを即時更新
        
        # 会話履歴保存ボタン
        if st.button("💾 会話履歴を保存", use_container_width=True):
            logger.info("会話履歴保存ボタンがクリックされました。")
            filename = save_conversation_to_html()
            if filename:
                st.success(f"会話履歴を保存しました: {filename}")
            else:
                st.error("会話履歴の保存に失敗しました。")
        
        # 手動更新ボタン
        if st.button("🔄 画面更新", use_container_width=True):
            logger.info("画面更新ボタンがクリックされました。")
            update_session_state()
            st.rerun()  # UIを即時更新
        
        # 自動更新の設定
        auto_refresh = st.checkbox("自動更新（1秒ごと）", value=True)
        
        # 更新インジケータ
        if _update_ui:
            st.success("新しい会話が追加されました！")
        
        st.divider()
        
        # 状態表示
        st.subheader("状態")
        st.write(f"録音中: {'はい' if st.session_state.is_listening else 'いいえ'}")
        st.write(f"応答生成中: {'はい' if st.session_state.is_generating else 'いいえ'}")
        st.write(f"最終更新: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time))}")
        
        # ログ表示
        st.subheader("ログ")
        if os.path.exists("app.log"):
            with open("app.log", "r") as f:
                log_content = f.read()
                st.text_area("最新のログ", log_content[-5000:], height=200)
        
        st.divider()
        
        # 使い方
        st.subheader("使い方")
        st.markdown("""
        1. 「録音開始」ボタンをクリックして音声認識を開始します。
        2. マイクに向かって話しかけてください。
        3. AIが自動的に会話を継続するか相槌を打つか判断します。
        4. 「録音停止」ボタンをクリックして音声認識を停止します。
        5. 「履歴クリア」ボタンをクリックして会話履歴をクリアします。
        6. 「会話履歴を保存」ボタンをクリックしてHTML形式で保存します。
        """)
        
        # ターン判定結果表示の切り替え（新規追加）
        show_turn_detection = st.checkbox("ターン判定結果を表示", value=True)
    
    # メインコンテンツ
    st.title("リアルタイム音声会話")
    
    # タブを使って表示を分ける
    tab1, tab2, tab3 = st.tabs(["🎤 リアルタイム会話", "🔄 ターン判定", "📝 会話履歴"])
    
    with tab1:
        # リアルタイム会話表示
        st.subheader("現在の会話")
        
        # 録音状態の表示
        if st.session_state.is_listening:
            st.info("🎤 録音中... マイクに向かって話しかけてください")
            
            # 最新のターン判定結果があれば表示
            if "turn_detection_results" in st.session_state and st.session_state.turn_detection_results:
                latest_result = st.session_state.turn_detection_results[-1]
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                        <strong>あなた (リアルタイム):</strong> {latest_result["transcript"]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # 相槌または応答を表示
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #e6f7ff; margin-bottom: 10px;">
                        <strong>AI (リアルタイム):</strong> {latest_result["acknowledgement"]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning("⏸️ 録音停止中")
        
        # 現在の文字起こしと応答を表示
        if st.session_state.current_transcript:
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                    <strong>あなた:</strong> {st.session_state.current_transcript}
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        if st.session_state.current_response:
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 5px; background-color: #e6f7ff; margin-bottom: 10px;">
                    <strong>AI:</strong> {st.session_state.current_response}
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with tab2:
        # ターン判定結果の表示
        st.subheader("ターン判定結果")
        
        # 録音中の場合はリアルタイム情報を表示
        if st.session_state.is_listening:
            st.info("🎤 リアルタイムターン判定中...")
        
        if "turn_detection_results" in st.session_state and st.session_state.turn_detection_results:
            for result in reversed(st.session_state.turn_detection_results):
                continue_val = result["continue_conversation"]
                color = "green" if continue_val else "red"
                icon = "🔄" if continue_val else "✅"
                
                # 最新の結果には「リアルタイム」マークを付ける
                is_latest = (result == st.session_state.turn_detection_results[-1])
                latest_mark = "⚡ 最新: " if is_latest else ""
                
                st.markdown(
                    f"""
                    <div style="margin-bottom:10px; padding:10px; border-left:4px solid {color}; background-color:rgba({0 if continue_val else 255}, {255 if continue_val else 0}, 0, 0.1)">
                        <strong>{icon} {latest_mark}{"会話継続" if continue_val else "会話完了"}:</strong> {result["transcript"]}
                        <br><small>相槌: "{result["acknowledgement"]}"</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("まだターン判定結果はありません")
    
    with tab3:
        # 会話履歴の表示
        st.subheader("会話履歴")
        
        if st.session_state.transcripts and st.session_state.responses:
            for i in range(min(len(st.session_state.transcripts), len(st.session_state.responses))):
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 5px;">
                        <strong>あなた:</strong> {st.session_state.transcripts[i]}
                    </div>
                    <div style="padding: 10px; border-radius: 5px; background-color: #e6f7ff; margin-bottom: 15px;">
                        <strong>AI:</strong> {st.session_state.responses[i]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.info("まだ会話履歴はありません")

    # UIの更新を処理するプレースホルダー
    ui_placeholder = st.empty()
    
    # 定期的な更新のためのカウンター
    if "update_counter" not in st.session_state:
        st.session_state.update_counter = 0
    
    # 定期的に更新（これはメインスレッドで実行される）
    st.session_state.update_counter += 1
    
    # グローバル変数からセッション状態に値をコピー
    if _current_transcript:
        st.session_state.current_transcript = _current_transcript
    if _current_response:
        st.session_state.current_response = _current_response

if __name__ == "__main__":
    logger.info("アプリケーションを起動します。")
    main() 