import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import base64
from openai import OpenAI

# utilsから必要な関数をimport
from utils.embedding_utils import (
    get_openai_client, get_embedding, load_embeddings, save_embeddings, compute_and_cache_faq_embeddings
)
from utils.chatbot_utils import find_top_similar, generate_response_with_history
from utils.logging_utils import append_to_csv, append_to_gsheet

# --- 設定 ---
st.set_page_config(page_title="LRADサポートチャット", layout="centered")
DATA_DIR = "data"
FAQ_PATH = os.path.join(DATA_DIR, "faq.csv")
EMBED_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
LOG_PATH = os.path.join(DATA_DIR, "chat_logs.csv")

# --- OpenAIクライアント ---
try:
    client = get_openai_client(st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI APIキーの取得に失敗しました。st.secretsの設定を確認してください。")
    st.stop()

# --- サイドバー設定 ---
st.sidebar.title("⚙️ 表示設定")
font_size = st.sidebar.selectbox("文字サイズを選んでください", ["小", "中", "大"])
font_size_map = {"小": "14px", "中": "18px", "大": "24px"}
img_width_map = {"小": 60, "中": 80, "大": 110}
selected_font = font_size_map[font_size]
selected_img = img_width_map[font_size]

max_log = st.sidebar.slider("チャット履歴の保存件数", min_value=10, max_value=200, value=100, step=10)
log_order = st.sidebar.radio("チャット履歴の表示順", ["新しい順", "古い順"])

# --- FAQ・埋め込みの読み込み ---
if not os.path.exists(FAQ_PATH):
    st.error("FAQデータがありません。data/faq.csvを用意してください。")
    st.stop()
faq_df = pd.read_csv(FAQ_PATH)
embeddings = compute_and_cache_faq_embeddings(faq_df, client, EMBED_PATH)

# --- FAQ埋め込み再計算 ---
if st.sidebar.button("FAQ埋め込み再計算"):
    embeddings = compute_and_cache_faq_embeddings(faq_df, client, EMBED_PATH)
    st.success("FAQ埋め込みを再計算しました。")
    st.experimental_rerun()

# --- CSS注入 ---
def inject_custom_css(selected_size):
    st.markdown(
        f"""
        <style>
        .chat-text, .stCaption, .css-ffhzg2 p, .stTextInput > label {{
            font-size: {selected_size} !important;
        }}
        .stTextInput > div > div > input {{
            font-size: {selected_size} !important;
        }}
        ::placeholder {{
            font-size: {selected_size} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
inject_custom_css(selected_font)

# --- ヘッダー画像 ---
def get_base64_image(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.warning(f"画像の読み込みに失敗しました: {e}")
        return ""

image_base64 = get_base64_image("LRADimg.png")
st.markdown(
    f"""
    <div style="display:flex; align-items:center;" class="chat-header">
        <img src="data:image/png;base64,{image_base64}"
             width="{selected_img}px" style="margin-right:10px;">
        <h1 style="margin:0; font-size:40px; font-weight:bold;">LRADサポートチャット</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

# --- よくある質問のランダム表示 ---
def display_random_faqs(faq_df, n=3):
    if len(faq_df) == 0:
        st.info("FAQがありません。")
        return
    sampled = faq_df.sample(min(n, len(faq_df)))
    for i, row in enumerate(sampled.itertuples(), 1):
        question = getattr(row, "質問", "（質問が不明です）")
        answer = getattr(row, "回答", "（回答が不明です）")
        st.markdown(
            f'<div class="chat-text"><b>❓ {question}</b><br>🅰️ {answer}</div><hr>',
            unsafe_allow_html=True
        )

st.markdown("### 💡 よくある質問（ランダム表示）")
display_random_faqs(faq_df, n=3)
st.divider()

# --- チャット履歴管理 ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# --- 入力フォーム ---
with st.form(key="chat_form", clear_on_submit=True):
    user_q = st.text_input("質問をどうぞ：")
    send = st.form_submit_button("送信")

# --- チャット送信処理 ---
if send and user_q:
    # 入力バリデーションはutils側でやってもOK
    ref_row = find_top_similar(user_q, faq_df, embeddings, client, get_embedding)
    if ref_row is None or ref_row.empty:
        answer = "申し訳ありません、関連FAQが見つかりませんでした。"
    else:
        ref_q = ref_row.iloc[0]["質問"]
        ref_a = ref_row.iloc[0]["回答"]
        with st.spinner("回答生成中…"):
            answer = generate_response_with_history(
                user_q,
                st.session_state.chat_log,
                ref_q,
                ref_a,
                client,
                max_turns=5
            )
    st.session_state.chat_log.insert(0, (user_q, answer))
    append_to_csv(user_q, answer, LOG_PATH)
    # GSheets保存は必要に応じて
    if "GoogleSheets" in st.secrets:
        append_to_gsheet(
            user_q, answer,
            st.secrets["GoogleSheets"]["sheet_key"],
            st.secrets["GoogleSheets"]["service_account_info"]
        )
    if len(st.session_state.chat_log) > max_log:
        st.session_state.chat_log = st.session_state.chat_log[:max_log]
    st.experimental_rerun()

# --- チャット履歴表示 ---
if st.session_state.chat_log:
    st.subheader("📜 チャット履歴")
    logs = st.session_state.chat_log if log_order == "新しい順" else list(reversed(st.session_state.chat_log))
    for q, a in logs:
        st.markdown(
            f'<div class="chat-text"><b>🧑‍💻 質問:</b> {q}<br><b>🤖 回答:</b> {a}</div><hr>',
            unsafe_allow_html=True
        )
