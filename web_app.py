"""
Speech-to-TextとLLMを連携させるFlaskベースのWebアプリケーション
"""
import os
import time
import threading
import queue
import json
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import pyaudio
import numpy as np
from dotenv import load_dotenv

from speech_to_text import SpeechToTextStreaming
from llm_manager import LLMManager

# 環境変数の読み込み
load_dotenv()

app = Flask(__name__)

# グローバル変数
stt = None
llm = None
is_listening = False
transcripts = []
responses = []
current_transcript = ""
transcript_queue = queue.Queue()
response_thread = None

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

def initialize_stt():
    """
    Speech-to-Textの初期化
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("環境変数GOOGLE_CLOUD_PROJECTが設定されていません。")
        return None
    
    try:
        return SpeechToTextStreaming(
            project_id=project_id,
            language_code="ja-JP",
            use_short_model=False
        )
    except Exception as e:
        print(f"Speech-to-Textの初期化中にエラーが発生しました: {str(e)}")
        return None

def initialize_llm():
    """
    LLMの初期化
    """
    try:
        return LLMManager()
    except Exception as e:
        print(f"LLMの初期化中にエラーが発生しました: {str(e)}")
        return None

def on_speech_result(transcript, is_final):
    """
    音声認識結果を受け取るコールバック関数
    
    Args:
        transcript: 認識されたテキスト
        is_final: 最終結果かどうか
    """
    global current_transcript
    
    # 現在の文字起こしを更新
    current_transcript = transcript
    
    # 最終結果の場合はキューに追加
    if is_final and transcript.strip():
        transcript_queue.put(transcript)
        transcripts.append(transcript)

def process_transcripts():
    """
    音声認識結果を処理するスレッド関数
    """
    global is_listening
    
    while is_listening:
        try:
            # キューから文字起こしを取得
            if not transcript_queue.empty():
                transcript = transcript_queue.get(timeout=0.1)
                
                # 文字数に基づいて相槌か会話かを判断
                if len(transcript) < 10:  # 短い発言は相槌
                    system_prompt = AIZUCHI_SYSTEM_PROMPT
                else:  # 長い発言は会話
                    system_prompt = CONVERSATION_SYSTEM_PROMPT
                
                # LLMで応答を生成
                response = llm.call_model(
                    prompt=transcript,
                    system_prompt=system_prompt,
                    model="gemini-2.0-flash"
                )
                
                # 応答を保存
                responses.append(response)
            
            # 少し待機
            time.sleep(0.1)
        except queue.Empty:
            # キューが空の場合は少し待機
            time.sleep(0.1)
        except Exception as e:
            print(f"文字起こしの処理中にエラーが発生しました: {str(e)}")
            break

def start_listening():
    """
    音声認識を開始する
    """
    global stt, llm, is_listening, response_thread
    
    if is_listening:
        return
    
    # Speech-to-Textの初期化
    if not stt:
        stt = initialize_stt()
    
    # LLMの初期化
    if not llm:
        llm = initialize_llm()
    
    if not stt or not llm:
        print("初期化に失敗しました。")
        return
    
    # 音声認識を開始
    stt.start_listening(callback=on_speech_result)
    is_listening = True
    
    # 文字起こし処理スレッドを開始
    response_thread = threading.Thread(target=process_transcripts)
    response_thread.daemon = True
    response_thread.start()
    
    print("音声認識を開始しました。")

def stop_listening():
    """
    音声認識を停止する
    """
    global stt, is_listening, response_thread
    
    if not is_listening:
        return
    
    # 音声認識を停止
    if stt:
        stt.stop_listening()
    
    is_listening = False
    
    # スレッドが終了するのを待機
    if response_thread and response_thread.is_alive():
        response_thread.join(timeout=1.0)
    
    print("音声認識を停止しました。")

def clear_history():
    """
    会話履歴をクリアする
    """
    global transcripts, responses, current_transcript
    
    transcripts.clear()
    responses.clear()
    current_transcript = ""
    
    # キューをクリア
    while not transcript_queue.empty():
        transcript_queue.get()

@app.route('/')
def index():
    """
    メインページを表示
    """
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def api_start():
    """
    音声認識を開始するAPI
    """
    start_listening()
    return jsonify({"status": "success", "message": "音声認識を開始しました。"})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """
    音声認識を停止するAPI
    """
    stop_listening()
    return jsonify({"status": "success", "message": "音声認識を停止しました。"})

@app.route('/api/clear', methods=['POST'])
def api_clear():
    """
    会話履歴をクリアするAPI
    """
    clear_history()
    return jsonify({"status": "success", "message": "会話履歴をクリアしました。"})

@app.route('/api/status')
def api_status():
    """
    現在の状態を取得するAPI
    """
    return jsonify({
        "is_listening": is_listening,
        "current_transcript": current_transcript
    })

@app.route('/api/history')
def api_history():
    """
    会話履歴を取得するAPI
    """
    history = []
    
    # 会話履歴を作成
    for i in range(max(len(transcripts), len(responses))):
        item = {}
        
        if i < len(transcripts):
            item["transcript"] = transcripts[i]
        
        if i < len(responses):
            item["response"] = responses[i]
        
        history.append(item)
    
    return jsonify({"history": history})

@app.route('/api/stream')
def api_stream():
    """
    SSEを使用して状態をストリーミングするAPI
    """
    def generate():
        last_transcript = ""
        last_response_count = 0
        
        while True:
            data = {
                "is_listening": is_listening,
                "current_transcript": current_transcript
            }
            
            # 新しい応答があれば追加
            if len(responses) > last_response_count:
                data["new_response"] = responses[-1]
                last_response_count = len(responses)
            
            # 文字起こしが変わったら更新
            if current_transcript != last_transcript:
                data["transcript_changed"] = True
                last_transcript = current_transcript
            
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)
    
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

# テンプレートディレクトリを作成
os.makedirs('templates', exist_ok=True)

# HTMLテンプレートを作成
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write("""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>リアルタイム音声会話</title>
    <style>
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            display: flex;
            gap: 20px;
        }
        
        .sidebar {
            flex: 1;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        .main-content {
            flex: 3;
            display: flex;
            gap: 20px;
        }
        
        .column {
            flex: 1;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        
        button.stop {
            background-color: #f44336;
        }
        
        button.stop:hover {
            background-color: #d32f2f;
        }
        
        button.clear {
            background-color: #2196F3;
        }
        
        button.clear:hover {
            background-color: #0b7dda;
        }
        
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f0f0;
        }
        
        .transcript-box, .response-box {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            min-height: 100px;
            margin-bottom: 20px;
            background-color: #fff;
        }
        
        .history-item {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        
        .user-text {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .ai-text {
            color: #2196F3;
            font-weight: bold;
        }
        
        hr {
            border: 0;
            height: 1px;
            background-color: #ddd;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <h1>🎤 リアルタイム音声会話</h1>
    
    <div class="container">
        <div class="sidebar">
            <h2>設定</h2>
            
            <button id="startBtn" onclick="startListening()">🎤 録音開始</button>
            <button id="stopBtn" class="stop" onclick="stopListening()" disabled>⏹️ 録音停止</button>
            <button id="clearBtn" class="clear" onclick="clearHistory()">🗑️ 履歴クリア</button>
            
            <div class="status">
                <h3>状態</h3>
                <p>録音中: <span id="listeningStatus">いいえ</span></p>
            </div>
            
            <hr>
            
            <h3>使い方</h3>
            <ol>
                <li>「録音開始」ボタンをクリックして音声認識を開始します。</li>
                <li>マイクに向かって話しかけてください。</li>
                <li>短い発言には相槌が、長い発言には会話が返されます。</li>
                <li>「録音停止」ボタンをクリックして音声認識を停止します。</li>
                <li>「履歴クリア」ボタンをクリックして会話履歴をクリアします。</li>
            </ol>
        </div>
        
        <div class="main-content">
            <div class="column">
                <h2>あなたの発言</h2>
                
                <h3>現在の文字起こし</h3>
                <div id="currentTranscript" class="transcript-box"></div>
                
                <h3>過去の発言</h3>
                <div id="transcriptHistory"></div>
            </div>
            
            <div class="column">
                <h2>AIの応答</h2>
                
                <h3>最新の応答</h3>
                <div id="currentResponse" class="response-box"></div>
                
                <h3>過去の応答</h3>
                <div id="responseHistory"></div>
            </div>
        </div>
    </div>
    
    <script>
        // グローバル変数
        let isListening = false;
        let eventSource = null;
        
        // ページ読み込み時に実行
        window.onload = function() {
            // 初期状態を取得
            fetchStatus();
            fetchHistory();
            
            // SSEを開始
            startEventSource();
        };
        
        // SSEを開始
        function startEventSource() {
            if (eventSource) {
                eventSource.close();
            }
            
            eventSource = new EventSource('/api/stream');
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // 状態を更新
                updateStatus(data);
                
                // 新しい応答があれば表示
                if (data.new_response) {
                    document.getElementById('currentResponse').textContent = data.new_response;
                    fetchHistory();
                }
                
                // 文字起こしが変わったら更新
                if (data.transcript_changed) {
                    document.getElementById('currentTranscript').textContent = data.current_transcript;
                }
            };
            
            eventSource.onerror = function() {
                console.error('SSE接続エラー');
                eventSource.close();
                
                // 5秒後に再接続
                setTimeout(startEventSource, 5000);
            };
        }
        
        // 状態を取得
        function fetchStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateStatus(data);
                })
                .catch(error => console.error('状態の取得に失敗しました:', error));
        }
        
        // 状態を更新
        function updateStatus(data) {
            isListening = data.is_listening;
            
            document.getElementById('listeningStatus').textContent = isListening ? 'はい' : 'いいえ';
            document.getElementById('startBtn').disabled = isListening;
            document.getElementById('stopBtn').disabled = !isListening;
            
            if (data.current_transcript) {
                document.getElementById('currentTranscript').textContent = data.current_transcript;
            }
        }
        
        // 履歴を取得
        function fetchHistory() {
            fetch('/api/history')
                .then(response => response.json())
                .then(data => {
                    updateHistory(data.history);
                })
                .catch(error => console.error('履歴の取得に失敗しました:', error));
        }
        
        // 履歴を更新
        function updateHistory(history) {
            const transcriptHistory = document.getElementById('transcriptHistory');
            const responseHistory = document.getElementById('responseHistory');
            
            // 履歴をクリア
            transcriptHistory.innerHTML = '';
            responseHistory.innerHTML = '';
            
            // 履歴を表示（新しいものから）
            history.slice().reverse().forEach(item => {
                if (item.transcript) {
                    const div = document.createElement('div');
                    div.className = 'history-item';
                    div.innerHTML = `<span class="user-text">あなた:</span> ${item.transcript}`;
                    transcriptHistory.appendChild(div);
                }
                
                if (item.response) {
                    const div = document.createElement('div');
                    div.className = 'history-item';
                    div.innerHTML = `<span class="ai-text">AI:</span> ${item.response}`;
                    responseHistory.appendChild(div);
                }
            });
            
            // 最新の応答を表示
            if (history.length > 0 && history[history.length - 1].response) {
                document.getElementById('currentResponse').textContent = history[history.length - 1].response;
            }
        }
        
        // 音声認識を開始
        function startListening() {
            fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                fetchStatus();
            })
            .catch(error => console.error('音声認識の開始に失敗しました:', error));
        }
        
        // 音声認識を停止
        function stopListening() {
            fetch('/api/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                fetchStatus();
            })
            .catch(error => console.error('音声認識の停止に失敗しました:', error));
        }
        
        // 履歴をクリア
        function clearHistory() {
            fetch('/api/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                document.getElementById('currentTranscript').textContent = '';
                document.getElementById('currentResponse').textContent = '';
                document.getElementById('transcriptHistory').innerHTML = '';
                document.getElementById('responseHistory').innerHTML = '';
            })
            .catch(error => console.error('履歴のクリアに失敗しました:', error));
        }
    </script>
</body>
</html>""")

if __name__ == '__main__':
    app.run(debug=True) 