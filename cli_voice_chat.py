"""
Speech-to-TextとLLMを連携させるCLIアプリケーション
"""
import os
import time
import threading
import queue
import logging
import json
import re
import signal
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv
import traceback

from speech_to_text_cli import SpeechToTextCLI as SpeechToTextStreaming
from llm_manager import LLMManager

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cli_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# 環境変数を明示的に設定
os.environ["GOOGLE_CLOUD_PROJECT"] = "livetoon-kaiwa"

# Google Cloud認証情報ファイルのパスを設定（必要に応じて）
credentials_path = os.path.join(os.path.dirname(__file__), "credentials.json")
if os.path.exists(credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# グローバル変数（スレッド間で共有）
_is_listening = False
_transcript_queue = queue.Queue()
_llm_manager = None
_transcripts = []
_responses = []
_current_transcript = ""
_current_response = ""
_is_generating = False
_accumulated_context = ""
_turn_detection_results = []

# ファイルベースの状態保存
_STATE_FILE = "cli_app_state.json"
_state_lock = threading.RLock()

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

# 会話応答用のシステムプロンプト
CONVERSATION_SYSTEM_PROMPT = """
あなたは親しみやすく自然な会話ができるAIアシスタントです。
ユーザーの質問や発言に対して、適切で自然な応答をしてください。
応答は簡潔で分かりやすく、親しみやすい口調を心がけてください。
"""

def parse_turn_decision(response_text, transcript):
    """
    LLMからのターン判定応答をパースする関数
    """
    try:
        # JSONブロックを抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # ```jsonなしの場合、直接JSONを探す
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
        
        # 余分な文字を削除
        json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)  # 非ASCII文字を削除
        
        # JSONをパース
        result = json.loads(json_str)
        
        # キーの正規化（camelCaseとsnake_caseの両方に対応）
        continue_conversation = result.get('continueConversation', result.get('continue_conversation', False))
        acknowledgement = result.get('acknowledgement', result.get('acknowledgment', "はい"))
        
        return continue_conversation, acknowledgement
    
    except Exception as e:
        logger.error(f"ターン判定の解析に失敗しました: {str(e)}")
        logger.error(f"解析対象テキスト: {response_text}")
        
        # バックアップヒューリスティック
        is_short = len(transcript) < 10
        has_question = any(q in transcript for q in ["?", "？", "何", "どう", "なぜ", "いつ", "どこ", "だれ", "誰", "ですか"])
        
        if has_question:
            return False, "ご質問にお答えします"
        elif is_short:
            return True, "はい"
        else:
            return False, "なるほど"

def save_state():
    """
    現在の状態をファイルに保存する
    """
    with _state_lock:
        state = {
            "transcripts": _transcripts,
            "responses": _responses,
            "turn_detection_results": _turn_detection_results
        }
        
        try:
            with open(_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.info(f"状態をファイルに保存しました。転記数: {len(_transcripts)}, 応答数: {len(_responses)}")
        except Exception as e:
            logger.error(f"状態の保存中にエラーが発生しました: {str(e)}")

def load_state():
    """
    保存された状態をファイルから読み込む
    """
    global _transcripts, _responses, _turn_detection_results
    
    if os.path.exists(_STATE_FILE):
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
                
            with _state_lock:
                _transcripts = state.get("transcripts", [])
                _responses = state.get("responses", [])
                _turn_detection_results = state.get("turn_detection_results", [])
                
            logger.info(f"ファイルから状態を読み込みました。転記数: {len(_transcripts)}, 応答数: {len(_responses)}")
        except Exception as e:
            logger.error(f"状態の読み込み中にエラーが発生しました: {str(e)}")
    else:
        logger.info("状態ファイルが見つかりません。新しい状態を作成します。")

def process_transcripts():
    """
    音声認識結果を処理するスレッド関数
    """
    global _is_listening, _transcript_queue, _llm_manager, _transcripts, _responses
    global _current_transcript, _current_response, _is_generating, _accumulated_context, _turn_detection_results
    
    logger.info("文字起こし処理スレッドを開始します。")
    
    while _is_listening:
        try:
            if not _transcript_queue.empty():
                transcript = _transcript_queue.get(timeout=0.1)
                logger.info(f"キューから取得した文字起こし: {transcript}")
                
                # グローバル変数に保存
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
                
                # 判定結果を保存
                turn_result = {
                    "transcript": transcript,
                    "continue_conversation": continue_conversation,
                    "acknowledgement": ack,
                    "raw_response": turn_response,
                    "timestamp": time.time()
                }
                
                with _state_lock:
                    _turn_detection_results.append(turn_result)
                    save_state()
                
                # ターミナルに表示
                print(f"\n\033[94mあなた: {transcript}\033[0m")
                
                if continue_conversation:
                    # 会話継続の場合は相槌を表示
                    print(f"\033[92mAI (相槌): {ack}\033[0m")
                    
                    # 短い発言を蓄積
                    _accumulated_context += f"{transcript} "
                else:
                    # 会話完了の場合はLLM応答を生成
                    print(f"\033[93mAI (応答生成中...)\033[0m")
                    logger.info(f"会話完了と判断: 応答生成開始")
                    _is_generating = True
                    
                    try:
                        # 会話履歴を構築
                        conversation_history = ""
                        for i in range(min(len(_transcripts), len(_responses))):
                            conversation_history += f"ユーザー: {_transcripts[i]}\nAI: {_responses[i]}\n"
                        
                        # 現在の会話コンテキストを追加
                        current_context = f"{conversation_history}ユーザー: {transcript}\nAI: "
                        
                        # LLM応答の生成
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
                            _current_response = response_text
                            save_state()
                        
                        # ターミナルに表示
                        print(f"\033[92mAI: {response_text}\033[0m")
                        
                    except Exception as e:
                        logger.error(f"LLM応答生成中にエラーが発生しました: {str(e)}")
                        logger.error(str(e))
                        traceback.print_exc()
                        print(f"\033[91mエラー: 応答生成に失敗しました\033[0m")
                    finally:
                        _is_generating = False
        except Exception as e:
            logger.error(f"文字起こしの処理中にエラーが発生しました: {str(e)}")
            logger.error(str(e))
            traceback.print_exc()
        
        time.sleep(0.1)  # CPUの使用率を下げるために短いスリープ

def on_transcript(transcript):
    """
    音声認識結果を受け取るコールバック関数
    """
    if transcript.strip():
        _transcript_queue.put(transcript)

def signal_handler(sig, frame):
    """
    Ctrl+Cで終了するためのシグナルハンドラ
    """
    global _is_listening
    print("\n\033[93m終了処理中...\033[0m")
    _is_listening = False
    time.sleep(1)  # スレッドが終了するのを待つ
    print("\033[92m終了しました\033[0m")
    sys.exit(0)

def print_history():
    """
    会話履歴を表示する
    """
    if not _transcripts or not _responses:
        print("\033[93mまだ会話履歴はありません\033[0m")
        return
    
    print("\n\033[1m===== 会話履歴 =====\033[0m")
    for i in range(min(len(_transcripts), len(_responses))):
        print(f"\033[94mあなた: {_transcripts[i]}\033[0m")
        print(f"\033[92mAI: {_responses[i]}\033[0m")
        print()

def main():
    """
    メイン関数
    """
    global _is_listening, _llm_manager
    
    # シグナルハンドラの設定
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\033[1m===== リアルタイム音声会話 CLI =====\033[0m")
    print("コマンド: [s]開始 [p]停止 [h]履歴表示 [c]履歴クリア [q]終了")
    
    # 状態の読み込み
    load_state()
    
    while True:
        cmd = input("\033[93mコマンド> \033[0m").strip().lower()
        
        if cmd == 's' or cmd == 'start':
            if _is_listening:
                print("\033[93m既に録音中です\033[0m")
                continue
            
            print("\033[92m音声認識を開始します...\033[0m")
            
            # Speech-to-Textの初期化
            try:
                logger.info("Speech-to-Textを初期化します。")
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "livetoon-kaiwa")  # デフォルト値を追加
                
                if not project_id:
                    raise ValueError("GOOGLE_CLOUD_PROJECTが設定されていません")
                
                logger.info(f"Speech-to-Textを初期化します。プロジェクトID: {project_id}")
                
                # SpeechToTextStreamingクラスの初期化方法を修正
                stt = SpeechToTextStreaming(
                    project_id=project_id
                )
                
                # コールバック関数を設定（クラスの実装に合わせて）
                stt.set_callback(on_transcript)
                
                logger.info("Speech-to-Textの初期化に成功しました。")
                
                # LLMの初期化
                logger.info("LLMを初期化します。")
                _llm_manager = LLMManager()
                logger.info("LLMの初期化に成功しました。")
                
                # 音声認識の開始
                _is_listening = True
                stt.start()
                logger.info("マイクからの音声認識を開始しました。")
                print("音声認識を開始しました。")
                
                # 文字起こし処理スレッドの開始
                transcript_thread = threading.Thread(target=process_transcripts)
                transcript_thread.daemon = True
                transcript_thread.start()
                logger.info("文字起こし処理スレッドを開始しました。")
                
            except Exception as e:
                logger.error(f"音声認識の開始中にエラーが発生しました: {str(e)}")
                print(f"\033[91mエラー: 音声認識の開始に失敗しました: {str(e)}\033[0m")
                traceback.print_exc()
                _is_listening = False
        
        elif cmd == 'p' or cmd == 'stop':
            if not _is_listening:
                print("\033[93m録音は既に停止しています\033[0m")
                continue
            
            print("\033[92m音声認識を停止します...\033[0m")
            _is_listening = False
            logger.info("音声認識を停止しました。")
        
        elif cmd == 'h' or cmd == 'history':
            print_history()
        
        elif cmd == 'c' or cmd == 'clear':
            with _state_lock:
                _transcripts = []
                _responses = []
                _turn_detection_results = []
                save_state()
            print("\033[92m会話履歴をクリアしました\033[0m")
            logger.info("会話履歴をクリアしました。")
        
        elif cmd == 'q' or cmd == 'quit':
            if _is_listening:
                print("\033[93m録音を停止しています...\033[0m")
                _is_listening = False
                time.sleep(1)  # スレッドが終了するのを待つ
            
            print("\033[92m終了します\033[0m")
            break
        
        else:
            print("\033[93m無効なコマンドです。[s]開始 [p]停止 [h]履歴表示 [c]履歴クリア [q]終了\033[0m")

if __name__ == "__main__":
    logger.info("CLIアプリケーションを起動します。")
    main() 