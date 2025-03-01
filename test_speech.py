"""
Google Cloud Speech-to-Text APIのテストスクリプト
"""
import os
import time
import json
import re
from dotenv import load_dotenv
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
import pyaudio
from llm_manager import LLMManager

# 環境変数の読み込み
load_dotenv()

# 音声設定
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
LANGUAGE_CODE = "ja-JP"  # 日本語

# LLMのシステムプロンプト
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
"""

CONVERSATION_SYSTEM_PROMPT = """
あなたは会話の相手です。ユーザーの発言に対して、自然な会話を続けるように返答してください。
質問には答え、意見には共感や別の視点を提供し、会話を発展させてください。
返答は簡潔で自然な会話調にしてください。
"""

# 新しい関数：LLMを初期化する
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

# 新しい関数：LLMを使用してターン判定を行う
def check_turn_taking(text, llm_manager):
    """
    LLMを使用してターン判定を行う
    
    Args:
        text: 判定するテキスト
        llm_manager: LLMマネージャー
        
    Returns:
        tuple: (会話継続か, 相槌や返事)
    """
    if not llm_manager:
        # LLMマネージャーがない場合はデフォルト値を返す
        return True, "はい"
    
    try:
        turn_response = llm_manager.call_model(
            prompt=text,
            system_prompt=TURN_DETECTION_PROMPT,
            model="gemini-2.0-flash",
            stream=False
        )
        print(f"ターン判定結果: {turn_response}")
        
        # JSONを解析
        try:
            # 正規表現でJSONを抽出（LLMが余計なテキストを含む場合）
            json_match = re.search(r'\{.*\}', turn_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                turn_data = json.loads(json_str)
                continue_conversation = turn_data.get("continueConversation", True)
                ack = turn_data.get("acknowledgement", "なるほど")
                return continue_conversation, ack
            else:
                # JSONが見つからない場合はデフォルト値を返す
                return True, "なるほど"
        except json.JSONDecodeError:
            # JSON解析エラーの場合はデフォルト値を返す
            return True, "なるほど"
    except Exception as e:
        print(f"ターン判定中にエラーが発生しました: {str(e)}")
        return True, "なるほど"

# test_speech_apiを拡張
def test_speech_api():
    """
    Speech-to-Text APIの接続テスト
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("環境変数GOOGLE_CLOUD_PROJECTが設定されていません。")
        return
    
    print(f"プロジェクトID: {project_id}")
    
    # LLMの初期化
    llm = initialize_llm()
    
    # 蓄積されたコンテキスト
    accumulated_context = ""
    
    try:
        # クライアントの初期化
        client = SpeechClient()
        print("SpeechClientの初期化に成功しました。")
        
        # 認識設定 - v2 APIでは異なる方法でエンコーディングを指定
        recognition_config = cloud_speech_types.RecognitionConfig(
            # v2 APIではexplicit_decoding_configを使用
            explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                audio_channel_count=1
            ),
            language_codes=[LANGUAGE_CODE],
            model="long",
        )
        
        print("認識設定の作成に成功しました。")
        
        # マイクのテスト
        try:
            audio = pyaudio.PyAudio()
            print("PyAudioの初期化に成功しました。")
            
            # 利用可能なマイクの一覧を表示
            print("\n利用可能なマイクデバイス:")
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"デバイス {i}: {device_info['name']}")
            
            # ターン判定をテストするためのモード選択
            print("\nテストモードを選択してください:")
            print("1. 基本的な音声認識テスト（5秒間録音）")
            print("2. リアルタイム音声認識とターン判定テスト")
            mode = input("選択 (1/2): ")
            
            if mode == "1":
                # 基本的な音声認識テスト
                # マイクストリームを開く
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )
                
                print("マイクストリームを開きました。5秒間録音します...")
                
                # 5秒間録音
                frames = []
                for i in range(0, int(RATE / CHUNK * 5)):
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    # 進捗表示
                    if i % 10 == 0:
                        print(".", end="", flush=True)
                
                print("\n録音完了")
                
                # ストリームを閉じる
                stream.stop_stream()
                stream.close()
                audio.terminate()
                
                # 録音データを結合
                audio_content = b''.join(frames)
                
                print(f"録音データのサイズ: {len(audio_content)} バイト")
                
                # 認識リクエストを作成
                request = cloud_speech_types.RecognizeRequest(
                    recognizer=f"projects/{project_id}/locations/global/recognizers/_",
                    config=recognition_config,
                    content=audio_content
                )
                
                print("音声認識リクエストを送信中...")
                
                # 音声認識を実行
                response = client.recognize(request=request)
                
                # 結果を表示
                print("\n認識結果:")
                for result in response.results:
                    for alternative in result.alternatives:
                        print(f"文字起こし: {alternative.transcript}")
                        print(f"信頼度: {alternative.confidence}")
                
                if not response.results:
                    print("認識結果がありません。")
            
            elif mode == "2":
                # リアルタイム音声認識とターン判定テスト
                # ストリーミング設定
                streaming_config = cloud_speech_types.StreamingRecognitionConfig(
                    config=recognition_config
                )
                
                # 設定リクエスト
                config_request = cloud_speech_types.StreamingRecognizeRequest(
                    recognizer=f"projects/{project_id}/locations/global/recognizers/_",
                    streaming_config=streaming_config,
                )
                
                # マイクストリームを開く
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )
                
                print("\nリアルタイム音声認識を開始します。終了するには Ctrl+C を押してください...")
                
                # 音声生成ジェネレータ
                def audio_generator():
                    while True:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        yield data
                
                # リクエスト生成ジェネレータ
                def request_generator():
                    # 最初に設定リクエストを送信
                    yield config_request
                    
                    # 音声データを送信
                    for audio_data in audio_generator():
                        request = cloud_speech_types.StreamingRecognizeRequest(audio=audio_data)
                        yield request
                
                # 認識結果の処理
                try:
                    responses = client.streaming_recognize(
                        requests=request_generator()
                    )
                    
                    for response in responses:
                        if not response.results:
                            continue
                            
                        # 最新の結果を取得
                        result = response.results[0]
                        
                        # 最終結果かどうか
                        is_final = result.is_final
                        
                        if result.alternatives:
                            # 認識されたテキスト
                            transcript = result.alternatives[0].transcript
                            
                            if is_final:
                                print(f"\n音声認識結果（最終）: {transcript}")
                                
                                # ターン判定
                                if llm:
                                    continue_conv, ack = check_turn_taking(transcript, llm)
                                    print(f"ターン判定: 会話継続={continue_conv}, 相槌=\"{ack}\"")
                                    
                                    if continue_conv:
                                        # ユーザーがまだ話している: コンテキストを蓄積して相槌
                                        accumulated_context += " " + transcript
                                        print(f"相槌を返します: {ack}")
                                        print(f"蓄積コンテキスト: {accumulated_context}")
                                    else:
                                        # ユーザーの発言が完了: フル応答を生成
                                        combined_prompt = accumulated_context + " " + transcript if accumulated_context else transcript
                                        print(f"完全な応答を生成します。入力=\"{combined_prompt}\"")
                                        if llm:
                                            response = llm.call_model(
                                                prompt=combined_prompt,
                                                system_prompt=CONVERSATION_SYSTEM_PROMPT,
                                                model="gemini-2.0-flash"
                                            )
                                            print(f"LLMの応答: {response}")
                                        accumulated_context = ""  # 蓄積コンテキストをクリア
                            else:
                                print(f"音声認識結果（中間）: {transcript}", end="\r")
                except KeyboardInterrupt:
                    print("\n\nリアルタイム音声認識を終了します...")
                finally:
                    # ストリームを閉じる
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
            
            else:
                print("無効な選択です。")
        
        except Exception as e:
            print(f"マイクのテスト中にエラーが発生しました: {str(e)}")
        
    except Exception as e:
        print(f"Speech-to-Text APIのテスト中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    print("Google Cloud Speech-to-Text APIのテストを開始します...")
    test_speech_api()
    print("テスト完了") 