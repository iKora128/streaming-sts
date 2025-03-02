"""
ターン判定テスター - 会話の継続/終了判定を視覚化するStreamlitツール
"""
import os
import re
import json
import streamlit as st
from dotenv import load_dotenv

# LLMマネージャーをインポート
from llm_manager import LLMManager

# 環境変数の読み込み
load_dotenv()

# ターン判定用のプロンプト
TURN_DETECTION_PROMPT = """
あなたはユーザーの発言を分析し、会話の流れを判断するアシスタントです。
ユーザーの発言が会話の途中であるか完了しているかを判断し、適切な応答方法を決定してください。

以下のJSON形式のみで出力してください。他の説明は一切不要です。
{
  "continueConversation": true または false,
  "acknowledgement": "適切な短い相槌や返事",
  "reason": "判断理由の短い説明"
}

- continueConversation: 
  - true: ユーザーの発言が一時停止しているが、まだ発言の途中で続きが予想される場合
  - false: ユーザーの発言が完結していて、AIが応答すべき場合

- acknowledgement:
  - continueConversationがtrueの場合: "なるほど", "はい", "ええ" などの非常に短い相槌
  - continueConversationがfalseの場合: ユーザーの発言に対する適切な応答の始まり

- reason:
  - この判断に至った理由の短い説明（デバッグ用）

例：
ユーザー：「今日は天気が」
出力：{"continueConversation": true, "acknowledgement": "はい", "reason": "文が途中で終わっている"}

ユーザー：「今日は天気が良いですね」
出力：{"continueConversation": false, "acknowledgement": "そうですね、とても良い天気です", "reason": "完結した文で意見が述べられている"}
"""

def backup_heuristic_analysis(text):
    """
    バックアップヒューリスティック分析（LLM解析が失敗した場合）
    """
    # 日本語用の簡易ヒューリスティック
    has_question_mark = "?" in text or "？" in text
    is_short = len(text.strip()) < 10
    likely_question = any(q in text for q in ["何", "どう", "なぜ", "いつ", "どこ", "だれ", "誰", "ですか"])
    
    # 未完了文のチェック
    incomplete_markers = ["は", "が", "けど", "って", "とか", "ので", "から"]
    ends_with_incomplete = any(text.strip().endswith(marker) for marker in incomplete_markers)
    
    # 文末表現のチェック
    ending_particles = ["ね", "よ", "な", "わ", "ぞ", "ぜ", "のだ", "んだ"]
    has_ending_particle = any(text.strip().endswith(particle) for particle in ending_particles)
    
    # 判断ロジック
    if has_question_mark or likely_question:
        continue_conversation = False
        reason = "質問文が検出されました"
    elif ends_with_incomplete:
        continue_conversation = True
        reason = "未完了の文が検出されました"
    elif has_ending_particle:
        continue_conversation = False
        reason = "文末表現が検出されました"
    elif is_short and not has_ending_particle:
        continue_conversation = True
        reason = "短い発言で文末表現がありません"
    else:
        # デフォルトは完了と判断
        continue_conversation = False
        reason = "デフォルト: 完了したと判断"
    
    # 適切な相槌の生成
    if continue_conversation:
        ack = "はい"
    else:
        ack = "なるほど、わかりました"
    
    return {
        "continueConversation": continue_conversation,
        "acknowledgement": ack,
        "reason": reason,
        "method": "バックアップヒューリスティック"
    }

def parse_llm_response(response_text):
    """
    LLM応答を解析してJSONデータを抽出（エラー処理改善版）
    """
    try:
        # まず直接JSONとして解析
        try:
            # 応答からJSONパターンを探す
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                data["method"] = "JSON正常解析"
                return data
        except json.JSONDecodeError:
            pass
        
        # 正規表現で個別にフィールドを抽出
        continue_match = re.search(r'"continueConversation"\s*:\s*(true|false)', response_text)
        ack_match = re.search(r'"acknowledgement"\s*:\s*"([^"]+)"', response_text)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', response_text)
        
        if continue_match and ack_match:
            continue_conversation = continue_match.group(1).lower() == "true"
            ack = ack_match.group(1)
            reason = reason_match.group(1) if reason_match else "理由が取得できませんでした"
            
            return {
                "continueConversation": continue_conversation,
                "acknowledgement": ack,
                "reason": reason,
                "method": "正規表現解析"
            }
            
        # すべての解析が失敗した場合、バックアップヒューリスティックを使用
        return backup_heuristic_analysis(response_text)
    
    except Exception as e:
        # それでも失敗した場合、バックアップヒューリスティックを使用
        st.error(f"解析エラー: {str(e)}")
        return backup_heuristic_analysis(response_text)

def main():
    st.set_page_config(
        page_title="ターン判定テスター",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 ターン判定テスター")
    st.subheader("会話継続判定の可視化ツール")
    
    # LLMの初期化
    if "llm" not in st.session_state:
        with st.spinner("LLMを初期化中..."):
            try:
                st.session_state.llm = LLMManager()
                st.success("LLMの初期化に成功しました。")
            except Exception as e:
                st.error(f"LLMの初期化中にエラーが発生しました: {str(e)}")
                st.stop()
    
    # サイドパネルにサンプル例を表示
    with st.sidebar:
        st.header("サンプル発言")
        
        examples = [
            "こんにちは",
            "今日の天気はどうですか？",
            "それが",
            "あのね",
            "最近忙しくて",
            "昨日映画を見に行きました",
            "え？",
            "なるほど、それで？",
            "人工知能について",
            "この前話した件について、あれはどうなりましたか？"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.user_input = example
                st.rerun()
    
    # メインコンテンツエリア
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("テスト入力")
        
        # テキスト入力
        user_input = st.text_area(
            "発言を入力してください",
            value=st.session_state.get("user_input", ""),
            height=100,
            key="input_area"
        )
        
        # モデル選択
        model = st.selectbox(
            "モデルを選択",
            ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash", "claude-3-sonnet"],
            index=0
        )
        
        # テストボタン
        if st.button("判定テスト", use_container_width=True):
            if not user_input.strip():
                st.warning("テキストを入力してください。")
            else:
                st.session_state.user_input = user_input  # 再利用のために保存
                with st.spinner("判定中..."):
                    # ターン判定用のLLM呼び出し
                    raw_response = st.session_state.llm.call_model(
                        prompt=user_input,
                        system_prompt=TURN_DETECTION_PROMPT,
                        model=model,
                        stream=False
                    )
                    
                    # 応答の解析
                    parsed_result = parse_llm_response(raw_response)
                    
                    # 比較用にバックアップヒューリスティックも実行
                    backup_result = backup_heuristic_analysis(user_input)
                    
                    # 結果をセッション状態に保存
                    st.session_state.raw_response = raw_response
                    st.session_state.parsed_result = parsed_result
                    st.session_state.backup_result = backup_result
                
                st.success("判定完了")
        
        # クリアボタン
        if st.button("クリア", use_container_width=True):
            if "user_input" in st.session_state:
                del st.session_state.user_input
            if "raw_response" in st.session_state:
                del st.session_state.raw_response
            if "parsed_result" in st.session_state:
                del st.session_state.parsed_result
            if "backup_result" in st.session_state:
                del st.session_state.backup_result
            st.rerun()
        
        # 詳細設定
        with st.expander("詳細設定"):
            st.text_area(
                "プロンプトをカスタマイズ",
                value=TURN_DETECTION_PROMPT,
                height=300,
                key="custom_prompt"
            )
    
    with col2:
        st.subheader("判定結果")
        
        if "parsed_result" in st.session_state:
            # 結果を視覚的に魅力的な方法で表示
            continue_conv = st.session_state.parsed_result.get("continueConversation", False)
            
            # カードスタイルの結果表示用CSS
            st.markdown("""
            <style>
            .result-card {
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .continue-true {
                background-color: #d1e7dd;
                border-left: 5px solid #198754;
            }
            .continue-false {
                background-color: #f8d7da;
                border-left: 5px solid #dc3545;
            }
            .comparison-card {
                background-color: #e2e3e5;
                border-left: 5px solid #6c757d;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # メイン結果
            st.markdown(
                f"""
                <div class="result-card {'continue-true' if continue_conv else 'continue-false'}">
                    <h3>{'✅ 会話継続' if continue_conv else '🛑 会話完了'}</h3>
                    <p><strong>相槌/応答:</strong> {st.session_state.parsed_result.get('acknowledgement', 'なし')}</p>
                    <p><strong>判断理由:</strong> {st.session_state.parsed_result.get('reason', 'なし')}</p>
                    <p><strong>解析方法:</strong> {st.session_state.parsed_result.get('method', 'なし')}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # バックアップヒューリスティックと比較
            with st.expander("ヒューリスティック比較"):
                backup_continue = st.session_state.backup_result.get("continueConversation", False)
                agreement = backup_continue == continue_conv
                
                st.markdown(
                    f"""
                    <div class="result-card comparison-card">
                        <h4>{'✓ 一致' if agreement else '✗ 不一致'}</h4>
                        <p><strong>ヒューリスティック判定:</strong> {'会話継続' if backup_continue else '会話完了'}</p>
                        <p><strong>理由:</strong> {st.session_state.backup_result.get('reason', 'なし')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # 生のLLM応答
            with st.expander("生の応答"):
                st.code(st.session_state.raw_response)
            
            # 履歴
            if "history" not in st.session_state:
                st.session_state.history = []
            
            # 現在の結果を履歴に追加（新しい場合）
            current_entry = {
                "input": st.session_state.user_input,
                "result": st.session_state.parsed_result
            }
            
            # 最新のエントリでない場合のみ追加
            if not st.session_state.history or st.session_state.history[-1]["input"] != current_entry["input"]:
                st.session_state.history.append(current_entry)
            
            # 履歴の表示
            if st.session_state.history:
                with st.expander("履歴", expanded=True):
                    for i, entry in enumerate(reversed(st.session_state.history[-10:])):
                        continue_val = entry["result"].get("continueConversation", False)
                        st.markdown(
                            f"""
                            <div style="margin-bottom:10px; padding:5px; border-left:3px solid {'green' if continue_val else 'red'}">
                                <strong>{"会話継続" if continue_val else "会話完了"}:</strong> {entry["input"]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.info("入力テキストを入力して「判定テスト」ボタンをクリックしてください。")
    
    # 使い方の説明
    st.markdown("---")
    st.markdown("""
    ### 使い方
    1. 左側のテキストエリアに発言を入力するか、サイドバーからサンプル発言を選択します。
    2. 「判定テスト」ボタンをクリックして、LLMによる判定結果を確認します。
    3. 右側に判定結果が表示されます。緑色は「会話継続」、赤色は「会話完了」を示します。
    4. 「履歴」セクションで過去の判定結果を確認できます。
    """)

if __name__ == "__main__":
    main() 