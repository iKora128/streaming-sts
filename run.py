"""
アプリケーションの実行スクリプト
"""
import os
import sys
import argparse
import subprocess

def check_requirements():
    """
    必要なパッケージがインストールされているか確認する
    """
    try:
        import streamlit
        import flask
        import google.cloud.speech_v2
        import pyaudio
        import dotenv
        import google.generativeai
        print("✅ 必要なパッケージがインストールされています。")
        return True
    except ImportError as e:
        print(f"❌ 必要なパッケージがインストールされていません: {e}")
        print("requirements.txtからパッケージをインストールしてください。")
        print("pip install -r requirements.txt")
        return False

def check_env_file():
    """
    .envファイルが存在するか確認する
    """
    if os.path.exists(".env"):
        print("✅ .envファイルが存在します。")
        return True
    else:
        print("❌ .envファイルが存在しません。")
        print(".envファイルを作成して必要な環境変数を設定してください。")
        return False

def run_streamlit():
    """
    Streamlitアプリケーションを実行する
    """
    print("🚀 Streamlitアプリケーションを起動しています...")
    subprocess.run(["streamlit", "run", "app.py"])

def run_flask():
    """
    Flaskアプリケーションを実行する
    """
    print("🚀 Flaskアプリケーションを起動しています...")
    subprocess.run(["python", "web_app.py"])

def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description="リアルタイム音声会話アプリケーションの実行")
    parser.add_argument("--web", action="store_true", help="Flaskベースのウェブアプリケーションを実行する")
    args = parser.parse_args()
    
    print("🎤 リアルタイム音声会話アプリケーション")
    print("=" * 50)
    
    # 必要なパッケージと環境変数を確認
    if not check_requirements() or not check_env_file():
        sys.exit(1)
    
    # アプリケーションを実行
    if args.web:
        run_flask()
    else:
        run_streamlit()

if __name__ == "__main__":
    main() 