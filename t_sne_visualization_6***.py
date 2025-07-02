import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.manifold import TSNE
import plotly.express as px

st.set_page_config(page_title="t-SNE 科目クラスタ表示", layout="wide")
st.title("🧠 t-SNE 可視化：歯科医師国家試験 Embedding（インタラクティブ表示）")

# ✅ 科目分類
subject_categories = [
    "解剖学", "組織学", "病理学", "生理学", "生化学", "微生物学", "薬理学", "歯科理工学", "疫学",
    "公衆衛生学", "口腔衛生学", "保存修復学", "歯内療法学", "歯周病学", "全部床義歯学", "部分床義歯学",
    "冠橋義歯学", "インプラント", "小児歯科学", "歯科矯正学", "口腔外科学", "歯科麻酔学",
    "歯科放射線学", "その他教養", "内科学", "高齢者"
]

def normalize_subject(raw_subject):
    for cat in subject_categories:
        if cat.replace("学", "") in raw_subject:
            return cat
    return "その他"

# ✅ データ読み込み
with open("embeddings_with_subject.pkl", "rb") as f:
    data = pickle.load(f)

if not isinstance(data, list) or not isinstance(data[0], tuple) or len(data[0]) != 2:
    st.error("❌ .pkl の中身が [(embedding, subject)] のタプルではありません。")
    st.stop()

embeddings = [item[0] for item in data]
raw_subjects = [str(item[1]) for item in data]
normalized_subjects = [normalize_subject(s) for s in raw_subjects]

# ✅ 問題文もCSVから読み込み（行順が揃っていればOK）
csv_df = pd.read_csv("8560sample.csv")
if "問題文" not in csv_df.columns:
    st.error("❌ CSV に '問題文' 列が見つかりません。")
    st.stop()
questions = csv_df["問題文"].astype(str).tolist()

# ✅ 科目選択
selected_category = st.selectbox("🔍 強調する科目を選択してください", subject_categories)

# ✅ t-SNE 実行
st.info("📌 t-SNE計算中…")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
reduced = tsne.fit_transform(np.array(embeddings))

# ✅ データフレームにまとめる
df_plot = pd.DataFrame({
    "x": reduced[:, 0],
    "y": reduced[:, 1],
    "subject": normalized_subjects,
    "question": questions
})

# ✅ 色分け
df_plot["color"] = np.where(
    df_plot["subject"] == selected_category, "red", "lightgray"
)

# ✅ 重心計算
centroid_all = df_plot[["x", "y"]].mean()
centroid_selected = df_plot[df_plot["subject"] == selected_category][["x", "y"]].mean()

# ✅ 重心データ
df_centroids = pd.DataFrame({
    "x": [centroid_all["x"], centroid_selected["x"]],
    "y": [centroid_all["y"], centroid_selected["y"]],
    "color": ["black", "blue"],
    "question": ["全体の重心", f"{selected_category} の重心"]
})

# ✅ 結合
df_all = pd.concat([df_plot, df_centroids], ignore_index=True)

# ✅ サイズ指定
df_all["size"] = np.where(df_all["color"].isin(["black", "blue"]), 14,
                   np.where(df_all["color"] == "red", 5, 3))

# ✅ プロット作成
fig = px.scatter(
    df_all,
    x="x", y="y",
    color="color",
    hover_data=["subject", "question"],
    size="size",
    opacity=0.8,
    color_discrete_map={
        "red": "red",
        "lightgray": "lightgray",
        "black": "black",
        "blue": "blue"
    }
)

# ✅ 重心を ×マークにする
for trace in fig.data:
    if trace.name in ["black", "blue"]:
        trace.marker.symbol = "x"

# ✅ 表示
st.markdown(f"📊 **t-SNE 科目別クラスタ表示：{selected_category} を強調**")
fig.update_layout(height=800)
st.plotly_chart(fig, use_container_width=False)