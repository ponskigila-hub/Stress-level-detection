import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Emotion & Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
EMOTION_CSV  = os.path.join(DATA_DIR, "emotion_accuracy_training.csv")
SENTIMENT_CSV = os.path.join(DATA_DIR, "ugm_fess_labeled.csv")

EMOTION_LABEL_MAP = {
    "anger": "😡 Anger",
    "happy": "😊 Happy",
    "sadness": "😢 Sadness",
    "fear": "😨 Fear",
    "love": "❤️ Love",
}
SENTIMENT_LABEL_MAP = {0: "😐 Netral", 1: "😊 Positif", 2: "😢 Negatif"}
SENTIMENT_COLOR_MAP = {0: "#9E9E9E", 1: "#4CAF50", 2: "#F44336"}

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+|#\w+", " ", text)               # mentions / hashtags
    text = re.sub(r"[^a-zA-Z\sÀ-ÿ]", " ", text)         # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_emotion_data():
    df = pd.read_csv(EMOTION_CSV)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["label", "tweet"])
    df["text_clean"] = df["tweet"].apply(clean_text)
    return df

@st.cache_data(show_spinner=False)
def load_sentiment_data():
    df = pd.read_csv(SENTIMENT_CSV)
    df.columns = df.columns.str.strip()
    label_col = [c for c in df.columns if "label" in c.lower()][0]
    df = df.rename(columns={label_col: "label_raw"})
    df["label"] = df["label_raw"].astype(str).str.extract(r"^(\d+)").astype(float)
    df = df.dropna(subset=["label", "full_text"])
    df["label"] = df["label"].astype(int)
    df["text_clean"] = df["full_text"].apply(clean_text)
    return df

# ── Model training ─────────────────────────────────────────────────────────────
def build_pipeline(model_name: str):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Linear SVM":          LinearSVC(max_iter=2000, C=1.0, random_state=42),
    }
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)),
        ("clf",   models[model_name]),
    ])

@st.cache_resource(show_spinner=False)
def train_model(task: str, model_name: str, test_size: float, _df):
    X = _df["text_clean"]
    y = _df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    model_path = os.path.join(MODELS_DIR, f"{task}_{model_name.replace(' ', '_')}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    return pipe, acc, f1, cm, report, y_test, y_pred

# ── Visualisation helpers ──────────────────────────────────────────────────────
def plot_label_distribution(df, label_col, label_map, title):
    counts = df[label_col].value_counts().sort_index()
    labels = [label_map.get(k, str(k)) for k in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Jumlah Sampel")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

def plot_class_metrics(report, class_names):
    rows = []
    for name in class_names:
        key = str(name)
        if key in report:
            rows.append({
                "Class": name, "Precision": report[key]["precision"],
                "Recall": report[key]["recall"], "F1-Score": report[key]["f1-score"],
            })
    metrics_df = pd.DataFrame(rows).set_index("Class")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics_df))
    w = 0.25
    ax.bar(x - w, metrics_df["Precision"], w, label="Precision", color="#42A5F5")
    ax.bar(x,     metrics_df["Recall"],    w, label="Recall",    color="#66BB6A")
    ax.bar(x + w, metrics_df["F1-Score"],  w, label="F1-Score",  color="#FFA726")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Konfigurasi")
task = st.sidebar.radio(
    "📌 Pilih Task",
    ["Emotion Detection", "Sentiment Analysis"],
    help="Emotion: 5 kelas emosi | Sentiment: 3 kelas sentimen UGM FESS",
)
model_name = st.sidebar.selectbox(
    "🤖 Pilih Model",
    ["Logistic Regression", "Naive Bayes", "Linear SVM"],
)
test_size = st.sidebar.slider(
    "🔀 Test Size (%)", min_value=10, max_value=40, value=20, step=5
) / 100

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset**\n"
    "- Emotion: 4 401 tweet berlabel emosi\n"
    "- Sentiment: ~31 800 post UGM FESS (0=Netral, 1=Positif, 2=Negatif)\n\n"
    "**Pipeline**: TF-IDF (bigram, 30k fitur) → Classifier"
)

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🧠 NLP Emotion & Sentiment Analyzer")
st.caption("Deteksi emosi dan sentimen teks Bahasa Indonesia menggunakan Machine Learning")

tab_eda, tab_train, tab_predict = st.tabs(["📊 Eksplorasi Data", "🏋️ Training & Evaluasi", "🔮 Prediksi"])

# ── Tab 1 : EDA ────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Eksplorasi Dataset")

    if task == "Emotion Detection":
        with st.spinner("Memuat data emosi…"):
            df = load_emotion_data()
        label_map = EMOTION_LABEL_MAP
        label_col = "label"
        text_col  = "tweet"
    else:
        with st.spinner("Memuat data sentimen…"):
            df = load_sentiment_data()
        label_map = SENTIMENT_LABEL_MAP
        label_col = "label"
        text_col  = "full_text"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sampel", f"{len(df):,}")
    c2.metric("Jumlah Kelas", df[label_col].nunique())
    c3.metric("Avg. Panjang Teks (kata)", int(df[text_col].str.split().str.len().mean()))

    st.markdown("#### Distribusi Label")
    fig_dist = plot_label_distribution(df, label_col, label_map, f"Distribusi Label – {task}")
    st.pyplot(fig_dist, use_container_width=True)

    st.markdown("#### Contoh Data")
    sample_label = st.selectbox(
        "Filter label",
        ["Semua"] + [label_map.get(k, str(k)) for k in sorted(df[label_col].unique())],
    )
    if sample_label == "Semua":
        show_df = df.sample(min(10, len(df)), random_state=1)
    else:
        inv_map = {v: k for k, v in label_map.items()}
        sel_key = inv_map.get(sample_label, sample_label)
        show_df = df[df[label_col] == sel_key].sample(min(10, (df[label_col] == sel_key).sum()), random_state=1)

    disp = show_df[[text_col, label_col]].copy()
    disp[label_col] = disp[label_col].map(lambda x: label_map.get(x, str(x)))
    st.dataframe(disp.rename(columns={text_col: "Teks", label_col: "Label"}), use_container_width=True, height=280)

# ── Tab 2 : Training ───────────────────────────────────────────────────────────
with tab_train:
    st.subheader(f"Training – {task} | Model: {model_name}")

    if task == "Emotion Detection":
        df_train = load_emotion_data()
        label_map = EMOTION_LABEL_MAP
        class_names = list(df_train["label"].unique())
    else:
        df_train = load_sentiment_data()
        label_map = SENTIMENT_LABEL_MAP
        class_names = sorted(df_train["label"].unique())

    if st.button("🚀 Mulai Training", use_container_width=True):
        with st.spinner(f"Training {model_name} pada {task}…"):
            pipe, acc, f1, cm, report, y_test, y_pred = train_model(
                task, model_name, test_size, df_train
            )

        st.success("✅ Training selesai!")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("F1-Score (weighted)", f"{f1:.4f}")

        col_cm, col_cls = st.columns(2)
        with col_cm:
            class_display = [label_map.get(k, str(k)) for k in sorted(set(y_test))]
            fig_cm = plot_confusion_matrix(cm, class_display, "Confusion Matrix")
            st.pyplot(fig_cm, use_container_width=True)
        with col_cls:
            fig_cls = plot_class_metrics(report, sorted(set(y_test)))
            st.pyplot(fig_cls, use_container_width=True)

        st.markdown("#### Classification Report")
        rep_rows = []
        for k, v in report.items():
            if isinstance(v, dict):
                rep_rows.append({"Class": label_map.get(k, k), **{m: round(v[m], 4) for m in ["precision", "recall", "f1-score", "support"]}})
        st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

        st.info(f"💾 Model disimpan di `{MODELS_DIR}/`")
    else:
        st.info("Klik tombol **Mulai Training** untuk melatih model dengan konfigurasi yang dipilih di sidebar.")

# ── Tab 3 : Predict ────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("🔮 Prediksi Teks Baru")

    model_path = os.path.join(MODELS_DIR, f"{task}_{model_name.replace(' ', '_')}.pkl")

    if not os.path.exists(model_path):
        st.warning("⚠️ Model belum dilatih. Silakan ke tab **Training & Evaluasi** dan latih model terlebih dahulu.")
    else:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        if task == "Emotion Detection":
            label_map = EMOTION_LABEL_MAP
        else:
            label_map = SENTIMENT_LABEL_MAP

        st.markdown("#### Prediksi Satu Teks")
        user_text = st.text_area(
            "Masukkan teks di sini:",
            placeholder="Contoh: Aku sangat bahagia hari ini karena mendapat kabar gembira!",
            height=100,
        )

        if st.button("🔍 Prediksi", use_container_width=True):
            if user_text.strip():
                cleaned = clean_text(user_text)
                pred = loaded_model.predict([cleaned])[0]
                label_display = label_map.get(pred, str(pred))
                st.success(f"**Hasil Prediksi:** {label_display}")

                # show probabilities if available
                if hasattr(loaded_model, "predict_proba"):
                    probs = loaded_model.predict_proba([cleaned])[0]
                    classes = loaded_model.classes_
                    prob_df = pd.DataFrame({
                        "Label": [label_map.get(c, str(c)) for c in classes],
                        "Probabilitas": probs,
                    }).sort_values("Probabilitas", ascending=False)
                    fig_prob, ax = plt.subplots(figsize=(6, 3))
                    colors_p = ["#1565C0" if c == pred else "#90CAF9" for c in classes]
                    bars = ax.barh(
                        [label_map.get(c, str(c)) for c in classes],
                        probs, color=colors_p
                    )
                    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
                    ax.set_xlim(0, 1.1)
                    ax.set_xlabel("Probabilitas")
                    ax.set_title("Distribusi Probabilitas", fontsize=11, fontweight="bold")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_prob, use_container_width=True)
            else:
                st.warning("Masukkan teks terlebih dahulu.")

        st.markdown("---")
        st.markdown("#### Prediksi Batch (CSV)")
        uploaded = st.file_uploader("Upload CSV dengan kolom `text`", type=["csv"])
        if uploaded:
            batch_df = pd.read_csv(uploaded)
            if "text" not in batch_df.columns:
                st.error("CSV harus memiliki kolom bernama **text**.")
            else:
                batch_df["text_clean"] = batch_df["text"].apply(clean_text)
                batch_df["prediksi"]   = loaded_model.predict(batch_df["text_clean"])
                batch_df["label"]      = batch_df["prediksi"].map(lambda x: label_map.get(x, str(x)))
                st.dataframe(batch_df[["text", "label"]], use_container_width=True)
                csv_out = batch_df[["text", "label"]].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Hasil", csv_out, "hasil_prediksi.csv", "text/csv")
