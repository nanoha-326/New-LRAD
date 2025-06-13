import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_faqs(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(np.stack(embeddings))
    return labels, kmeans

def get_faq_stats(faq_df):
    return {
        "total": len(faq_df),
        "categories": faq_df["カテゴリ"].value_counts().to_dict() if "カテゴリ" in faq_df.columns else {}
    }

def plot_cluster_scatter(embeddings, labels):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.stack(embeddings))
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10")
    plt.title("FAQクラスタ可視化")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    return plt
