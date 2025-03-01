"""
Google Cloud Speech-to-Text APIを使用してストリーミング音声認識を行うモジュール
"""
import os
import queue
import threading
import time
from typing import Callable, List, Optional

import pyaudio
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# 音声設定
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
LANGUAGE_CODE = "ja-JP"  # 日本語

class SpeechToTextStreaming:
    """
    Google Cloud Speech-to-Text APIを使用してストリーミング音声認識を行うクラス
    """
    
    def __init__(self, 
                project_id: Optional[str] = None, 
                language_code: str = LANGUAGE_CODE,
                use_short_model: bool = False):
        """
        SpeechToTextStreamingの初期化
        
        Args:
            project_id: Google CloudプロジェクトのプロジェクトID
            language_code: 言語コード
            use_short_model: 短い音声用のモデルを使用するかどうか
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("プロジェクトIDが設定されていません。環境変数GOOGLE_CLOUD_PROJECTを設定するか、引数で指定してください。")
            
        self.language_code = language_code
        self.use_short_model = use_short_model
        
        # クライアントの初期化
        self.client = SpeechClient()
        
        # 音声認識の設定
        self.streaming_config = None
        self.config_request = None
        self._setup_recognition_config()
        
        # 音声入力用の変数
        self.audio_interface = None
        self.audio_stream = None
        
        # 音声認識結果を保存するキュー
        self.result_queue = queue.Queue()
        
        # 音声認識の状態
        self.is_listening = False
        self.listen_thread = None
        
    def _setup_recognition_config(self):
        """
        音声認識の設定を行う
        """
        # モデル選択
        model = "latest_short" if self.use_short_model else "long"
        
        # 認識設定
        recognition_config = cloud_speech_types.RecognitionConfig(
            explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                audio_channel_count=1
            ),
            language_codes=[self.language_code],
            model=model,
        )
        
        # ストリーミング設定
        self.streaming_config = cloud_speech_types.StreamingRecognitionConfig(
            config=recognition_config
        )
        
        # 設定リクエスト
        self.config_request = cloud_speech_types.StreamingRecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
            streaming_config=self.streaming_config,
        )
    
    def _audio_generator(self):
        """
        マイクからの音声入力を生成するジェネレータ
        
        Yields:
            bytes: 音声データのチャンク
        """
        while self.is_listening:
            # マイクから音声データを取得
            data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
            yield data
    
    def _request_generator(self):
        """
        音声認識リクエストを生成するジェネレータ
        
        Yields:
            StreamingRecognizeRequest: 音声認識リクエスト
        """
        # 最初に設定リクエストを送信
        yield self.config_request
        
        # 音声データを送信
        for audio_data in self._audio_generator():
            request = cloud_speech_types.StreamingRecognizeRequest(audio=audio_data)
            yield request
    
    def _process_responses(self, responses, callback: Optional[Callable[[str, bool], None]] = None):
        """
        音声認識の応答を処理する
        
        Args:
            responses: 音声認識の応答
            callback: 認識結果を受け取るコールバック関数
        """
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
                
                # キューに追加
                self.result_queue.put((transcript, is_final))
                
                # コールバック関数がある場合は呼び出す
                if callback:
                    callback(transcript, is_final)
    
    def start_listening(self, callback: Optional[Callable[[str, bool], None]] = None):
        """
        マイクからの音声入力の認識を開始する
        
        Args:
            callback: 認識結果を受け取るコールバック関数
        """
        if self.is_listening:
            print("すでに音声認識を開始しています。")
            return
            
        # PyAudioの初期化
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        
        # 音声認識の状態を更新
        self.is_listening = True
        
        # 音声認識を別スレッドで実行
        def listen_thread_func():
            try:
                # 音声認識リクエストを送信
                responses = self.client.streaming_recognize(
                    requests=self._request_generator()
                )
                
                # 応答を処理
                self._process_responses(responses, callback)
            except Exception as e:
                print(f"音声認識中にエラーが発生しました: {str(e)}")
            finally:
                self.stop_listening()
        
        self.listen_thread = threading.Thread(target=listen_thread_func)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        print("音声認識を開始しました。")
    
    def stop_listening(self):
        """
        マイクからの音声入力の認識を停止する
        """
        if not self.is_listening:
            return
            
        # 音声認識の状態を更新
        self.is_listening = False
        
        # 音声ストリームを閉じる
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            
        # PyAudioを終了
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None
            
        print("音声認識を停止しました。")
    
    def get_result(self, block: bool = True, timeout: Optional[float] = None) -> Optional[tuple]:
        """
        音声認識の結果を取得する
        
        Args:
            block: キューが空の場合にブロックするかどうか
            timeout: タイムアウト時間（秒）
            
        Returns:
            tuple: (認識されたテキスト, 最終結果かどうか)
        """
        try:
            return self.result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_results(self):
        """
        音声認識の結果をクリアする
        """
        while not self.result_queue.empty():
            self.result_queue.get()
    
    def transcribe_file(self, 
                       file_path: str, 
                       callback: Optional[Callable[[str, bool], None]] = None) -> List[str]:
        """
        音声ファイルを文字起こしする
        
        Args:
            file_path: 音声ファイルのパス
            callback: 認識結果を受け取るコールバック関数
            
        Returns:
            List[str]: 認識結果のリスト
        """
        # 音声ファイルを読み込む
        with open(file_path, "rb") as f:
            audio_content = f.read()
            
        # チャンクに分割
        chunk_length = len(audio_content) // 5
        stream = [
            audio_content[start : start + chunk_length]
            for start in range(0, len(audio_content), chunk_length)
        ]
        
        # 音声認識リクエストを作成
        audio_requests = (
            cloud_speech_types.StreamingRecognizeRequest(audio=audio) for audio in stream
        )
        
        # リクエストを送信
        def requests():
            yield self.config_request
            yield from audio_requests
            
        # 音声認識を実行
        responses = self.client.streaming_recognize(requests=requests())
        
        # 結果を保存
        results = []
        
        # 応答を処理
        for response in responses:
            for result in response.results:
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    results.append(transcript)
                    
                    # コールバック関数がある場合は呼び出す
                    if callback:
                        callback(transcript, result.is_final)
        
        return results 