"""
CLI用にカスタマイズしたSpeech-to-Textクラス
"""
import os
import queue
import threading
import time
from google.cloud import speech
from google.cloud.speech_v1 import SpeechClient
from google.auth import credentials
from google.auth.transport import requests

class SpeechToTextCLI:
    """
    Google Cloud Speech-to-Textを使用した音声認識クラス（CLI用）
    """
    def __init__(self, project_id=None, callback=None):
        self.project_id = project_id
        self.callback = callback
        self.client = None
        self.streaming_config = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.transcript_thread = None
    
    def set_callback(self, callback):
        """コールバック関数を設定"""
        self.callback = callback
    
    def start(self):
        """音声認識を開始"""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        # Speech-to-Text クライアントの初期化（プロジェクトIDを明示的に指定）
        client_options = {"api_endpoint": "speech.googleapis.com:443"}
        if self.project_id:
            client_options["quota_project_id"] = self.project_id
        
        self.client = SpeechClient(client_options=client_options)
        
        # 設定
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ja-JP",
            enable_automatic_punctuation=True,
            model="latest_long",
        )
        
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )
        
        # マイク入力スレッドの開始
        self.mic_thread = threading.Thread(target=self._mic_thread)
        self.mic_thread.daemon = True
        self.mic_thread.start()
        
        # 認識スレッドの開始
        self.transcript_thread = threading.Thread(target=self._transcript_thread)
        self.transcript_thread.daemon = True
        self.transcript_thread.start()
    
    def stop(self):
        """音声認識を停止"""
        self.is_listening = False
    
    def _mic_thread(self):
        """マイクからの音声入力を処理するスレッド"""
        import pyaudio
        
        RATE = 16000
        CHUNK = int(RATE / 10)  # 100ms
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        while self.is_listening:
            data = stream.read(CHUNK, exception_on_overflow=False)
            self.audio_queue.put(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def _transcript_thread(self):
        """音声認識結果を処理するスレッド"""
        def request_generator():
            while self.is_listening:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    continue
        
        try:
            responses = self.client.streaming_recognize(
                config=self.streaming_config,
                requests=request_generator()
            )
            
            for response in responses:
                if not response.results:
                    continue
                
                result = response.results[0]
                if not result.alternatives:
                    continue
                
                transcript = result.alternatives[0].transcript
                
                if result.is_final:
                    if self.callback:
                        self.callback(transcript)
        except Exception as e:
            print(f"音声認識エラー: {str(e)}") 