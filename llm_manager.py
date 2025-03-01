"""
LLMモデルを管理するためのモジュール
"""
import os
from typing import Dict, List, Optional, Any, Callable
import json
from datetime import datetime

from google import genai
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

class LLMManager:
    """
    複数のLLMモデルを管理し、会話を実行するためのクラス
    """
    
    def __init__(self):
        """
        LLMManagerの初期化
        """
        # API Keyの設定
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # クライアントの初期化
        if self.google_api_key:
            self.genai_client = genai.Client(api_key=self.google_api_key)
        else:
            self.genai_client = None
        
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            
        if self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        else:
            self.anthropic_client = None
        
        # 利用可能なモデル
        self.available_models = {
            "gemini-2.0-flash-lite": self._call_gemini_new,
            "gemini-1.5-pro": self._call_gemini_new,
            "gemini-1.5-flash": self._call_gemini_new,
            "gpt-4o": self._call_openai,
            "claude-3-opus": self._call_anthropic,
            "claude-3-sonnet": self._call_anthropic,
        }
        
        # デフォルトのモデル設定
        self.assistant_model = os.getenv("ASSISTANT_MODEL", "gemini-2.0-flash-lite")
        self.human_model = os.getenv("HUMAN_MODEL", "gpt-4o")
        
        # 会話履歴
        self.conversation_history = []
        
    def _call_gemini(self, 
                    prompt: str, 
                    system_prompt: Optional[str] = None, 
                    model: str = "gemini-1.5-pro",
                    stream: bool = False,
                    stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Geminiモデル（1.5シリーズ）を呼び出す
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト
            model: モデル名
            stream: ストリーミングモードを使用するかどうか
            stream_callback: ストリーミング時のコールバック関数
            
        Returns:
            生成されたテキスト
        """
        if not self.google_api_key:
            return "Google API Keyが設定されていません。"
        
        try:
            generation_config = {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            # モデルの初期化
            genai_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # システムプロンプトがある場合は追加
            if system_prompt:
                chat = genai_model.start_chat(system_instruction=system_prompt)
                
                if stream:
                    response = chat.send_message(prompt, stream=True)
                    
                    # ストリーミング処理
                    full_response = ""
                    for chunk in response:
                        if hasattr(chunk, "text"):
                            if stream_callback:
                                stream_callback(chunk.text)
                            full_response += chunk.text
                    
                    return full_response
                else:
                    response = chat.send_message(prompt)
                    return response.text
            else:
                if stream:
                    response = genai_model.generate_content(prompt, stream=True)
                    
                    # ストリーミング処理
                    full_response = ""
                    for chunk in response:
                        if hasattr(chunk, "text"):
                            if stream_callback:
                                stream_callback(chunk.text)
                            full_response += chunk.text
                    
                    return full_response
                else:
                    response = genai_model.generate_content(prompt)
                    return response.text
                
        except Exception as e:
            return f"Geminiモデルの呼び出し中にエラーが発生しました: {str(e)}"
    
    def _call_gemini_new(self, 
                    prompt: str, 
                    system_prompt: Optional[str] = None, 
                    model: str = "gemini-2.0-flash-lite",
                    stream: bool = False,
                    stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Geminiモデルを呼び出す（新しいgenaiライブラリを使用）
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト
            model: モデル名
            stream: ストリーミングモードを使用するかどうか
            stream_callback: ストリーミング時のコールバック関数
            
        Returns:
            生成されたテキスト
        """
        if not self.google_api_key:
            return "Google API Keyが設定されていません。"
        
        try:
            # システムプロンプトがある場合は、configに含める
            contents = prompt
            config = None
            
            if system_prompt:
                config = genai.types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            
            if stream:
                response = self.genai_client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config
                )
                
                # ストリーミング処理
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, "text"):
                        if stream_callback:
                            stream_callback(chunk.text)
                        full_response += chunk.text
                
                return full_response
            else:
                response = self.genai_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                return response.text
                
        except Exception as e:
            return f"Geminiモデルの呼び出し中にエラーが発生しました: {str(e)}"
    
    def _call_openai(self, 
                    prompt: str, 
                    system_prompt: Optional[str] = None, 
                    model: str = "gpt-4o",
                    stream: bool = False,
                    stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        OpenAIモデルを呼び出す
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト
            model: モデル名
            stream: ストリーミングモードを使用するかどうか
            stream_callback: ストリーミング時のコールバック関数
            
        Returns:
            生成されたテキスト
        """
        if not self.openai_client:
            return "OpenAI API Keyが設定されていません。"
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            if stream:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.9,
                    max_tokens=2048,
                    stream=True
                )
                
                full_response = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        if stream_callback:
                            stream_callback(content)
                        full_response += content
                return full_response
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.9,
                    max_tokens=2048
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f"OpenAIモデルの呼び出し中にエラーが発生しました: {str(e)}"
    
    def _call_anthropic(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None, 
                       model: str = "claude-3-sonnet",
                       stream: bool = False,
                       stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Anthropicモデルを呼び出す
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト
            model: モデル名
            stream: ストリーミングモードを使用するかどうか
            stream_callback: ストリーミング時のコールバック関数
            
        Returns:
            生成されたテキスト
        """
        if not self.anthropic_client:
            return "Anthropic API Keyが設定されていません。"
        
        try:
            if stream:
                with self.anthropic_client.messages.stream(
                    model=model,
                    max_tokens=2048,
                    system=system_prompt if system_prompt else None,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.9,
                ) as stream:
                    full_response = ""
                    for text in stream.text_stream:
                        if stream_callback:
                            stream_callback(text)
                        full_response += text
                    return full_response
            else:
                message = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=system_prompt if system_prompt else None,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.9,
                )
                return message.content[0].text
                
        except Exception as e:
            return f"Anthropicモデルの呼び出し中にエラーが発生しました: {str(e)}"
    
    def call_model(self, 
                  prompt: str, 
                  system_prompt: Optional[str] = None, 
                  model: Optional[str] = None,
                  stream: bool = False,
                  stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        指定されたモデルを呼び出す
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト
            model: モデル名（指定しない場合はデフォルトモデルを使用）
            stream: ストリーミングモードを使用するかどうか
            stream_callback: ストリーミング時のコールバック関数
            
        Returns:
            生成されたテキスト
        """
        if not model:
            model = self.assistant_model
            
        if model not in self.available_models:
            return f"モデル '{model}' は利用できません。利用可能なモデル: {', '.join(self.available_models.keys())}"
            
        return self.available_models[model](
            prompt=prompt, 
            system_prompt=system_prompt, 
            model=model,
            stream=stream,
            stream_callback=stream_callback
        )
    
    def simulate_conversation(self, 
                             initial_prompt: str, 
                             assistant_system_prompt: str,
                             human_system_prompt: str,
                             num_turns: int = 5,
                             stream: bool = False,
                             stream_callback: Optional[Dict[str, Callable[[str], None]]] = None) -> List[Dict[str, Any]]:
        """
        LLM同士の会話をシミュレートする
        
        Args:
            initial_prompt: 初期プロンプト
            assistant_system_prompt: アシスタント用のシステムプロンプト
            human_system_prompt: 人間役用のシステムプロンプト
            num_turns: 会話のターン数
            stream: ストリーミングモードを使用するかどうか
            stream_callback: ストリーミング時のコールバック関数（"assistant"と"human"をキーとする辞書）
            
        Returns:
            会話履歴
        """
        conversation = []
        
        # 初期メッセージを追加
        conversation.append({
            "role": "human",
            "content": initial_prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        current_prompt = initial_prompt
        
        for i in range(num_turns):
            # アシスタントの応答
            assistant_callback = stream_callback.get("assistant") if stream_callback else None
            assistant_response = self.call_model(
                prompt=current_prompt,
                system_prompt=assistant_system_prompt,
                model=self.assistant_model,
                stream=stream,
                stream_callback=assistant_callback
            )
            
            conversation.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # 人間役の応答
            human_callback = stream_callback.get("human") if stream_callback else None
            human_response = self.call_model(
                prompt=f"以下はAIアシスタントからのメッセージです。あなたは人間として自然に返信してください。\n\n{assistant_response}",
                system_prompt=human_system_prompt,
                model=self.human_model,
                stream=stream,
                stream_callback=human_callback
            )
            
            conversation.append({
                "role": "human",
                "content": human_response,
                "timestamp": datetime.now().isoformat()
            })
            
            current_prompt = human_response
        
        self.conversation_history = conversation
        return conversation
    
    def save_conversation(self, filename: str = None) -> str:
        """
        会話履歴をJSONファイルに保存する
        
        Args:
            filename: 保存するファイル名（指定しない場合は現在の日時を使用）
            
        Returns:
            保存したファイルのパス
        """
        if not self.conversation_history:
            return "会話履歴がありません。"
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations/conversation_{timestamp}.json"
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            
        return filename
    
    def generate_html(self, filename: str = None) -> str:
        """
        会話履歴をHTMLファイルに変換する
        
        Args:
            filename: 保存するファイル名（指定しない場合は現在の日時を使用）
            
        Returns:
            保存したファイルのパス
        """
        if not self.conversation_history:
            return "会話履歴がありません。"
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations/conversation_{timestamp}.html"
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        html_content = """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LLM会話履歴</title>
            <style>
                body {
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .chat-container {
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                }
                .message {
                    margin-bottom: 15px;
                    padding: 10px 15px;
                    border-radius: 18px;
                    max-width: 80%;
                    position: relative;
                }
                .human {
                    background-color: #e1ffc7;
                    margin-left: auto;
                    border-bottom-right-radius: 0;
                }
                .assistant {
                    background-color: #f0f0f0;
                    margin-right: auto;
                    border-bottom-left-radius: 0;
                }
                .timestamp {
                    font-size: 0.7em;
                    color: #999;
                    margin-top: 5px;
                    text-align: right;
                }
                h1 {
                    text-align: center;
                    color: #2c3e50;
                }
                .role-label {
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .human .role-label {
                    color: #4caf50;
                }
                .assistant .role-label {
                    color: #2196f3;
                }
            </style>
        </head>
        <body>
            <h1>LLM会話履歴</h1>
            <div class="chat-container">
        """
        
        for message in self.conversation_history:
            role = message["role"]
            content = message["content"]
            timestamp = datetime.fromisoformat(message["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            
            role_label = "人間" if role == "human" else "アシスタント"
            
            # 改行をHTMLの<br>タグに置換
            content_html = content.replace('\n', '<br>')
            
            html_content += f"""
                <div class="message {role}">
                    <div class="role-label">{role_label}</div>
                    <div class="content">{content_html}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return filename 