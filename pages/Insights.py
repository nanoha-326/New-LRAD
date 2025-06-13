import streamlit as st
import pandas as pd
import os
from utils.logging_utils import load_chat_logs
from utils.embedding_utils import load_embeddings
from utils.analysis_utils import cluster_faqs, get_faq_stats, plot_cluster_scatter

st.set_page_config(page_title="Insights - LRADチャットボット", layout="wide")

st.title("📊 チャットボット利用・FAQ分析 Insights")

# データパス
DATA_DIR = "data"
FAQ_PATH = os.path.join(DATA_DIR, "faq.csv")
EMBED_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
LOG_PATH = os.path.join(DATA_DIR, "chat_logs.csv")

# チャットログ読み込み
chat_logs = load_chat_logs(LOG_PATH)
st.header("1. チャットログ分析")

if chat_logs.empty:
    st.info("チャットログがありません。")
else:
    st.write(f"総チャット件数: {len(chat_logs)}")
    # 日別件数
    chat_logs["date"] = pd.to_datetime(chat_logs["timestamp"]).dt.date
    daily_counts = chat_logs.groupby("date").size()
    st.line_chart(daily_counts, use_container_width=True)
    # 最近のやりとり
    st.subheader("直近のやりとり")
    st.dataframe(chat_logs.tail(10)[["timestamp", "question", "answer"]].rename(
        columns={"timestamp": "日時", "question": "質問", "answer": "回答"}
    ))

# FAQ分析
st.header("2. FAQ分析")
if not os.path.exists(FAQ_PATH):
    st.warning("FAQデータが見つかりません。")
else:
    faq_df = pd.read_csv(FAQ_PATH)
    stats = get_faq_stats(faq_df)
    st.write(f"FAQ総数: {stats['total']}")
    if stats["categories"]:
        st.write("カテゴリ分布:")
        st.bar_chart(pd.Series(stats["categories"]))

# FAQクラスタ可視化
st.header("3. FAQクラスタ可視化")
if not os.path.exists(EMBED_PATH):
    st.info("埋め込みデータがありません。")
else:
    embeddings = load_embeddings(EMBED_PATH)
    if embeddings is not None and len(embeddings) > 0:
        n_clusters = st.slider("クラスタ数", min_value=2, max_value=10, value=5)
        labels, kmeans = cluster_faqs(embeddings, n_clusters=n_clusters)
        st.write(f"クラスタごとのFAQ数: {pd.Series(labels).value_counts().sort_index().to_dict()}")
        plt = plot_cluster_scatter(embeddings, labels)
        st.pyplot(plt)
    else:
        st.info("埋め込みデータが空です。")

st.caption("※このページはチャットボットの利用状況やFAQの傾向を可視化します。")
