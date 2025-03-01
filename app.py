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

def save_state_to_file():
    """
    グローバル変数の状態をファイルに保存する
    """
    global _transcripts, _responses, _current_transcript, _current_response
    
    state = {
        "transcripts": _transcripts,
        "responses": _responses,
        "current_transcript": _current_transcript,
        "current_response": _current_response,
        "timestamp": time.time()
    }
    
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            import json
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"状態をファイルに保存しました。転記数: {len(_transcripts)}, 応答数: {len(_responses)}")
    except Exception as e:
        logger.error(f"状態のファイル保存中にエラーが発生しました: {str(e)}")

def load_state_from_file():
    """
    ファイルからグローバル変数の状態を読み込む
    """
    global _transcripts, _responses, _current_transcript, _current_response
    
    if not os.path.exists(_STATE_FILE):
        logger.info("状態ファイルが存在しません。")
        return False
    
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            import json
            state = json.load(f)
            
        _transcripts = state.get("transcripts", [])
        _responses = state.get("responses", [])
        _current_transcript = state.get("current_transcript", "")
        _current_response = state.get("current_response", "")
        
        logger.info(f"ファイルから状態を読み込みました。転記数: {len(_transcripts)}, 応答数: {len(_responses)}")
        return True
    except Exception as e:
        logger.error(f"状態のファイル読み込み中にエラーが発生しました: {str(e)}")
        return False

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
                transcript = _transcript_queue.get(timeout=0.1)
                logger.info(f"キューから取得した文字起こし: {transcript}")
                
                # テキスト長や質問形式に基づく簡易判定（バックアップとして）
                has_question_mark = "?" in transcript or "？" in transcript
                is_short = len(transcript.strip()) < 10
                likely_question = any(q in transcript for q in ["何", "どう", "なぜ", "いつ", "どこ", "だれ", "誰", "ですか"])
                
                # Call LLM to decide turn-taking
                turn_response = _llm_manager.call_model(
                    prompt=transcript,
                    system_prompt=TURN_DETECTION_PROMPT,
                    model="gemini-2.0-flash-lite",  # より軽量なモデルを使用
                    stream=False
                )
                logger.info(f"ターン判定結果: {turn_response}")
                
                # JSONパース処理の改善
                continue_conversation = True  # デフォルト値
                ack = "なるほど"  # デフォルト値
                
                try:
                    # 正規表現でJSONを抽出
                    json_match = re.search(r'\{.*?\}', turn_response.replace('\n', ' '), re.DOTALL)
                    if json_match:
                        try:
                            turn_data = json.loads(json_match.group(0))
                            continue_conversation = turn_data.get("continueConversation", True)
                            ack = turn_data.get("acknowledgement", "なるほど")
                        except json.JSONDecodeError:
                            # JSON解析に失敗した場合はバックアップロジックを使用
                            continue_conversation = not (has_question_mark or likely_question)
                            
                    else:
                        # JSONが見つからない場合はバックアップロジックを使用
                        continue_conversation = not (has_question_mark or likely_question)
                        if "acknowledgement" in turn_response:
                            ack_match = re.search(r'"acknowledgement":\s*"([^"]+)"', turn_response)
                            if ack_match:
                                ack = ack_match.group(1)
                except Exception as e:
                    # エラーが発生した場合はバックアップロジックを使用
                    logger.error(f"ターン判定の解析中にエラー: {str(e)}")
                    continue_conversation = not (has_question_mark or likely_question)
                
                # バックアップロジックによる追加チェック
                if likely_question or has_question_mark:
                    # 質問の形式が明確な場合は、常にfalseに強制
                    continue_conversation = False
                    logger.info(f"バックアップロジックにより会話継続をFalseに設定（質問検出）")
                
                # 結果をログに記録
                logger.info(f"最終判定: 会話継続={continue_conversation}, 相槌=\"{ack}\"")
                
                if continue_conversation:
                    # ユーザーが継続中: 短い相槌を返し、コンテキストを蓄積
                    _accumulated_context += " " + transcript
                    _responses.append(ack)
                    _transcripts.append(transcript)
                    
                    # 相槌のストリーミングシミュレーション
                    _current_response = ""  # 現在の応答をリセット
                    for char in ack:
                        _current_response += char  # 文字を追加
                        on_llm_stream(char)
                        time.sleep(0.01)  # 文字ごとに若干の遅延
                    
                    logger.info(f"会話継続中: 蓄積内容=\"{_accumulated_context}\"")
                else:
                    # ユーザーの発言が完了: 完全な応答を生成
                    combined_prompt = _accumulated_context + " " + transcript if _accumulated_context.strip() else transcript
                    logger.info(f"会話完了: 完全な応答を生成します。入力=\"{combined_prompt}\"")
                    
                    # LLM応答生成の進行状況を示すフラグを設定
                    _is_generating = True
                    _current_response = ""  # 現在の応答をリセット
                    
                    # ストリーミングを使用してLLMを呼び出し
                    _llm_manager.call_model(
                        prompt=combined_prompt,
                        system_prompt=CONVERSATION_SYSTEM_PROMPT,
                        model="gemini-2.0-flash-lite",
                        stream=True,
                        stream_callback=on_llm_stream
                    )
                    
                    # 注：ストリーミングの場合、応答はon_llm_streamコールバックを通じて処理され、
                    # その結果を_current_responseに蓄積し、最終的に_responsesに追加する
                    _responses.append(_current_response)
                    _transcripts.append(combined_prompt)
                    _accumulated_context = ""  # 蓄積コンテキストをクリア
                
                # ステータスフラグを更新
                _is_generating = False
                _update_ui = True
                _force_update = True
                _last_ui_update_time = time.time()
                save_state_to_file()
            
            time.sleep(0.1)  # ポーリング間隔
            
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"文字起こしの処理中にエラーが発生しました: {str(e)}")
            logger.exception(e)  # スタックトレースを記録
            time.sleep(1)  # エラー発生時は少し待機
    
    logger.info("文字起こし処理スレッドを終了します。")

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
    save_state_to_file()
    
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
    load_state_from_file()
    
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
    global _transcripts, _responses, _current_transcript, _current_response, _update_ui, _last_ui_update_time, _force_update, _is_generating
    
    logger.info("アプリケーションを開始します。")
    
    st.set_page_config(
        page_title="リアルタイム音声会話",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 リアルタイム音声会話")
    
    # ファイルから状態を読み込む
    load_state_from_file()
    
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
    
    # 自動更新のためのプレースホルダー
    update_placeholder = st.empty()
    
    # ファイルの更新時刻を確認
    try:
        file_modified_time = os.path.getmtime(_STATE_FILE) if os.path.exists(_STATE_FILE) else 0
    except:
        file_modified_time = 0
    
    # ファイルが更新されている場合はセッション状態を更新
    if file_modified_time > st.session_state.last_update_time:
        logger.info("ファイルが更新されているため、セッション状態を更新します。")
        load_state_from_file()
        update_session_state()
    
    # グローバル変数からセッション状態を更新（UIの更新フラグがセットされている場合）
    current_time = time.time()
    if _update_ui or _force_update or (auto_refresh and current_time - st.session_state.last_update_time > 1):
        update_session_state()
        _update_ui = False
        
        # 強制更新フラグがセットされている場合または自動更新が有効で最後のUI更新から3秒以上経過している場合は再読み込み
        if _force_update or (auto_refresh and current_time - _last_ui_update_time < 10 and _last_ui_update_time > st.session_state.last_update_time):
            logger.info("UIを再読み込みします。")
            time.sleep(0.1)
            st.rerun()
    
    # デバッグ情報
    st.sidebar.subheader("デバッグ情報")
    st.sidebar.write(f"グローバル転記数: {len(_transcripts)}")
    st.sidebar.write(f"グローバル応答数: {len(_responses)}")
    st.sidebar.write(f"セッション転記数: {len(st.session_state.transcripts)}")
    st.sidebar.write(f"セッション応答数: {len(st.session_state.responses)}")
    if os.path.exists(_STATE_FILE):
        st.sidebar.write(f"状態ファイル更新: {time.strftime('%H:%M:%S', time.localtime(file_modified_time))}")
    
    # メイン画面 - 吹き出し形式の会話表示
    st.header("会話履歴")
    
    # CSSスタイルを追加
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin-bottom: 20px;
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
        border: 1px solid #e5e5e5;
        border-bottom-left-radius: 5px;
    }
    .timestamp {
        font-size: 0.7em;
        color: #999;
        margin-top: 5px;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 現在の文字起こしと応答（リアルタイム表示）
    chat_container = st.container()
    
    with chat_container:
        # 現在の文字起こしがある場合は表示
        if st.session_state.current_transcript:
            # HTMLエスケープ処理
            current_transcript_escaped = st.session_state.current_transcript.replace("<", "&lt;").replace(">", "&gt;")
            
            st.markdown(f"""
            <div class="chat-container">
                <div class="message user-message">
                    <div class="message-bubble user-bubble">
                        <div>{current_transcript_escaped}</div>
                        <div class="timestamp">現在の発言</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 応答生成中の場合は表示
        if st.session_state.is_generating and st.session_state.current_response:
            # HTMLエスケープ処理
            current_response_escaped = st.session_state.current_response.replace("<", "&lt;").replace(">", "&gt;")
            
            st.markdown(f"""
            <div class="chat-container">
                <div class="message ai-message">
                    <div class="message-bubble ai-bubble">
                        <div>{current_response_escaped}</div>
                        <div class="timestamp">生成中...</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 過去の会話履歴を表示
        if st.session_state.transcripts and st.session_state.responses:
            chat_history = ""
            for i in range(len(st.session_state.transcripts) - 1, -1, -1):
                # HTMLエスケープ処理
                transcript_escaped = st.session_state.transcripts[i].replace("<", "&lt;").replace(">", "&gt;")
                
                # ユーザーメッセージ
                chat_history += f"""
                <div class="message user-message">
                    <div class="message-bubble user-bubble">
                        <div>{transcript_escaped}</div>
                    </div>
                </div>
                """
                
                # AIメッセージ
                if i < len(st.session_state.responses):
                    # HTMLエスケープ処理
                    response_escaped = st.session_state.responses[i].replace("<", "&lt;").replace(">", "&gt;")
                    
                    chat_history += f"""
                    <div class="message ai-message">
                        <div class="message-bubble ai-bubble">
                            <div>{response_escaped}</div>
                        </div>
                    </div>
                    """
            
            st.markdown(f'<div class="chat-container">{chat_history}</div>', unsafe_allow_html=True)
        else:
            st.info("会話履歴はまだありません。マイクに向かって話しかけてください。")
    
    # 自動更新の処理
    if auto_refresh:
        with update_placeholder:
            # ファイルの更新時刻を確認
            try:
                current_file_modified_time = os.path.getmtime(_STATE_FILE) if os.path.exists(_STATE_FILE) else 0
            except:
                current_file_modified_time = 0
            
            # 応答生成中または最後の更新から1秒以上経過している場合は再読み込み
            if st.session_state.is_generating or current_file_modified_time > file_modified_time or time.time() - st.session_state.last_update_time > 1:
                st.session_state.last_update_time = time.time()
                time.sleep(0.1)
                st.rerun()

if __name__ == "__main__":
    logger.info("アプリケーションを起動します。")
    main() 