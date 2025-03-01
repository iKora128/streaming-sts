"""
LLMと音声認識の機能をStreamlitなしでテストするスクリプト
"""
import os
import time
import threading
import queue
from typing import List, Optional

from dotenv import load_dotenv
from speech_to_text import SpeechToTextStreaming
from llm_manager import LLMManager

# 環境変数の読み込み
load_dotenv()

# グローバル変数
is_listening = False
transcript_queue = queue.Queue()
transcripts = []
responses = []
current_transcript = ""
_accumulated_context = ""

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

TURN_DETECTION_PROMPT = """
あなたはユーザーの発言を分析し、会話の流れを判断するアシスタントです。
ユーザーの発言が会話の途中であるか完了しているかを判断し、適切な応答方法を決定してください。

以下のJSON形式のみで出力してください。他の説明は一切不要です。
{
  "continueConversation": true または false,
  "acknowledgement": "適切な短い相槌や返事"
}

- continueConversation: 
  - true: ユーザーの発言が一時停止しているが、まだ発言の途中で続きが予想される場合
  - false: ユーザーの発言が完結していて、AIが応答すべき場合

- acknowledgement:
  - continueConversationがtrueの場合: "なるほど", "はい", "ええ" などの非常に短い相槌
  - continueConversationがfalseの場合: ユーザーの発言に対する適切な応答

例：
ユーザー：「今日は天気が」
出力：{"continueConversation": true, "acknowledgement": "はい"}

ユーザー：「今日は天気が良いですね」
出力：{"continueConversation": false, "acknowledgement": "そうですね、とても良い天気です"}
"""

def initialize_stt():
    """
    Speech-to-Textの初期化
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("環境変数GOOGLE_CLOUD_PROJECTが設定されていません。")
        return None
    
    try:
        print(f"Speech-to-Textを初期化します。プロジェクトID: {project_id}")
        stt = SpeechToTextStreaming(
            project_id=project_id,
            language_code="ja-JP",
            use_short_model=False
        )
        print("Speech-to-Textの初期化に成功しました。")
        return stt
    except Exception as e:
        print(f"Speech-to-Textの初期化中にエラーが発生しました: {str(e)}")
        return None

def initialize_llm():
    """
    LLMの初期化
    """
    try:
        print("LLMを初期化します。")
        llm = LLMManager()
        print("LLMの初期化に成功しました。")
        return llm
    except Exception as e:
        print(f"LLMの初期化中にエラーが発生しました: {str(e)}")
        return None

def on_speech_result(transcript: str, is_final: bool):
    """
    音声認識結果を受け取るコールバック関数
    
    Args:
        transcript: 認識されたテキスト
        is_final: 最終結果かどうか
    """
    global transcript_queue, current_transcript
    
    # 現在の文字起こしを更新
    current_transcript = transcript
    
    # ログ出力
    if is_final:
        print(f"音声認識結果（最終）: {transcript}")
    else:
        print(f"音声認識結果（中間）: {transcript}", end="\r")
    
    # 最終結果の場合はキューに追加
    if is_final and transcript.strip():
        transcript_queue.put(transcript)

def process_transcripts(llm_manager):
    """
    音声認識結果を処理するスレッド関数
    """
    global is_listening, transcript_queue, transcripts, responses, _accumulated_context
    
    print("文字起こし処理スレッドを開始します。")
    import json
    import re
    
    while is_listening:
        try:
            # キューから文字起こしを取得
            if not transcript_queue.empty():
                transcript = transcript_queue.get(timeout=0.1)
                print(f"キューから取得した文字起こし: {transcript}")
                
                # Call LLM to decide turn-taking using TURN_DETECTION_PROMPT
                turn_response = llm_manager.call_model(
                    prompt=transcript,
                    system_prompt=TURN_DETECTION_PROMPT,
                    model="gemini-2.0-flash",
                    stream=False
                )
                print(f"ターン判定結果: {turn_response}")
                
                # Parse JSON from turn_response
                try:
                    # Extract JSON using regex in case LLM includes other text
                    json_match = re.search(r'\{.*\}', turn_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        turn_data = json.loads(json_str)
                        continue_conversation = turn_data.get("continueConversation", False)
                        ack = turn_data.get("acknowledgement", "なるほど")
                    else:
                        # Fallback if no JSON found
                        continue_conversation = "continueConversation\": true" in turn_response
                        ack = "なるほど"
                        if "acknowledgement\":" in turn_response:
                            ack_match = re.search(r'acknowledgement\":\s*\"([^\"]+)', turn_response)
                            if ack_match:
                                ack = ack_match.group(1)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    continue_conversation = "continueConversation\": true" in turn_response
                    ack = "なるほど"
                    if "acknowledgement\":" in turn_response:
                        ack_match = re.search(r'acknowledgement\":\s*\"([^\"]+)', turn_response)
                        if ack_match:
                            ack = ack_match.group(1)
                
                print(f"解析結果: 会話継続={continue_conversation}, 相槌=\"{ack}\"")
                
                if continue_conversation:
                    # User is still talking: accumulate transcript and provide acknowledgement
                    _accumulated_context += " " + transcript
                    responses.append(ack)
                    transcripts.append(transcript)
                    # Simulate streaming response by printing each character with slight delay
                    print("相槌を返します: ", end="", flush=True)
                    for char in ack:
                        print(char, end="", flush=True)
                        time.sleep(0.01)
                    print()  # new line after ack
                    print(f"蓄積内容: {_accumulated_context}")
                else:
                    # User has completed their turn: generate full response
                    combined_prompt = _accumulated_context + " " + transcript if _accumulated_context else transcript
                    print(f"完全な応答を生成します: 入力=\"{combined_prompt}\"")
                    full_response = llm_manager.call_model(
                        prompt=combined_prompt,
                        system_prompt=CONVERSATION_SYSTEM_PROMPT,
                        model="gemini-2.0-flash",
                        stream=True,
                        stream_callback=lambda chunk: (print(chunk, end="", flush=True), time.sleep(0.01))
                    )
                    print()  # new line after response
                    responses.append(full_response)
                    transcripts.append(combined_prompt)
                    _accumulated_context = ""  # clear accumulated context
                
                # Display conversation history
                print("\n===== 会話履歴 =====")
                for i, (t, r) in enumerate(zip(transcripts, responses)):
                    print(f"あなた: {t}")
                    print(f"AI: {r}")
                    if i < len(transcripts) - 1:
                        print("-" * 30)
                print("=" * 20)
            
            # 少し待機
            time.sleep(0.1)
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"文字起こしの処理中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            break
    
    print("文字起こし処理スレッドを終了します。")

def start_listening(stt, llm):
    """
    音声認識を開始する
    
    Args:
        stt: Speech-to-Textインスタンス
        llm: LLMマネージャー
    """
    global is_listening
    
    if is_listening:
        print("すでに音声認識を開始しています。")
        return
    
    print("音声認識を開始します。")
    
    if not stt or not llm:
        print("初期化に失敗しました。")
        return
    
    # 音声認識を開始
    print("マイクからの音声認識を開始します。")
    stt.start_listening(callback=on_speech_result)
    is_listening = True
    
    # 文字起こし処理スレッドを開始
    print("文字起こし処理スレッドを開始します。")
    response_thread = threading.Thread(target=process_transcripts, args=(llm,))
    response_thread.daemon = True
    response_thread.start()
    
    return response_thread

def stop_listening(stt, response_thread):
    """
    音声認識を停止する
    
    Args:
        stt: Speech-to-Textインスタンス
        response_thread: 文字起こし処理スレッド
    """
    global is_listening
    
    if not is_listening:
        print("音声認識はすでに停止しています。")
        return
    
    print("音声認識を停止します。")
    
    # 音声認識を停止
    if stt:
        stt.stop_listening()
    
    is_listening = False
    
    # スレッドが終了するのを待機
    if response_thread and response_thread.is_alive():
        print("文字起こし処理スレッドの終了を待機します。")
        response_thread.join(timeout=1.0)
    
    print("音声認識を停止しました。")

def clear_history():
    """
    会話履歴をクリアする
    """
    global transcript_queue, transcripts, responses, current_transcript, _accumulated_context
    
    print("会話履歴をクリアします。")
    
    transcripts.clear()
    responses.clear()
    current_transcript = ""
    _accumulated_context = ""  # 蓄積コンテキストもクリア
    
    # キューをクリア
    while not transcript_queue.empty():
        transcript_queue.get()
    
    print("会話履歴をクリアしました。")

def test_llm_only():
    """
    LLMのみをテストする関数
    """
    print("\n===== LLMのみのテスト =====")
    
    # LLMの初期化
    llm = initialize_llm()
    if not llm:
        print("LLMの初期化に失敗しました。")
        return
    
    # テスト用のプロンプト
    test_prompts = [
        "こんにちは",
        "今日の天気はどうですか？",
        "人工知能について教えてください",
        "はい"
    ]
    
    # 各プロンプトでテスト
    for prompt in test_prompts:
        print(f"\nプロンプト: {prompt}")
        
        # 文字数に基づいて相槌か会話かを判断
        if len(prompt) < 10:
            system_prompt = AIZUCHI_SYSTEM_PROMPT
            print("短い発言のため、相槌モードを使用します。")
        else:
            system_prompt = CONVERSATION_SYSTEM_PROMPT
            print("長い発言のため、会話モードを使用します。")
        
        # LLMで応答を生成
        print("LLMで応答を生成します...")
        response = llm.call_model(
            prompt=prompt,
            system_prompt=system_prompt,
            model="gemini-2.0-flash"
        )
        print(f"LLMの応答: {response}")
    
    print("\nLLMのテストが完了しました。")

def test_speech_to_text():
    """
    Speech-to-Textのみをテストする関数
    """
    print("\n===== Speech-to-Textのみのテスト =====")
    
    # Speech-to-Textの初期化
    stt = initialize_stt()
    if not stt:
        print("Speech-to-Textの初期化に失敗しました。")
        return
    
    # 音声認識を開始
    print("マイクからの音声認識を開始します。5秒間録音します...")
    
    # コールバック関数
    results = []
    def callback(transcript, is_final):
        if is_final:
            print(f"音声認識結果（最終）: {transcript}")
            results.append(transcript)
        else:
            print(f"音声認識結果（中間）: {transcript}", end="\r")
    
    # 音声認識を開始
    stt.start_listening(callback=callback)
    
    # 5秒間待機
    time.sleep(5)
    
    # 音声認識を停止
    stt.stop_listening()
    print("音声認識を停止しました。")
    
    # 結果を表示
    print("\n認識結果:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result}")
    
    print("\nSpeech-to-Textのテストが完了しました。")

def main():
    """
    メイン関数
    """
    print("LLMと音声認識のテストを開始します。")
    
    while True:
        print("\n===== メニュー =====")
        print("1. LLMのみをテスト")
        print("2. Speech-to-Textのみをテスト")
        print("3. 音声認識とLLMの連携をテスト")
        print("0. 終了")
        
        choice = input("選択してください (0-3): ")
        
        if choice == "0":
            print("テストを終了します。")
            break
        
        elif choice == "1":
            test_llm_only()
        
        elif choice == "2":
            test_speech_to_text()
        
        elif choice == "3":
            # Speech-to-Textの初期化
            stt = initialize_stt()
            if not stt:
                print("Speech-to-Textの初期化に失敗しました。")
                continue
            
            # LLMの初期化
            llm = initialize_llm()
            if not llm:
                print("LLMの初期化に失敗しました。")
                continue
            
            # 音声認識を開始
            response_thread = start_listening(stt, llm)
            
            print("\n音声認識を開始しました。話しかけてください。")
            print("終了するには 'q' を入力してください。")
            
            # ユーザー入力を待機
            while True:
                cmd = input("")
                if cmd.lower() == 'q':
                    break
                elif cmd.lower() == 'c':
                    clear_history()
            
            # 音声認識を停止
            stop_listening(stt, response_thread)
        
        else:
            print("無効な選択です。もう一度選択してください。")

if __name__ == "__main__":
    main() 