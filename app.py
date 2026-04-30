import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import joblib

from datasets import load_dataset
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Stress Detection NLP",
    page_icon="🧠",
    layout="wide"
)

# ======================================
# CSS
# ======================================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #111827;
}
h1,h2,h3{
    color:#2563EB;
}
.stButton>button{
    width:100%;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# SESSION
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "model" not in st.session_state:
    st.session_state.model = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "results" not in st.session_state:
    st.session_state.results = None

stemmer = PorterStemmer()

# ======================================
# CLEAN TEXT
# ======================================
def clean_text(text):
    def clean_text(text):
    text = str(text).lower()

    # remove url
    text = re.sub(r"http\S+", "", text)

    # remove mention
    text = re.sub(r"@\w+", "", text)

    # remove hashtag
    text = re.sub(r"#\w+", "", text)

    # remove symbol/number
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # simple tokenize
    tokens = text.split()

    # stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)

# ======================================
# NORMALIZE DATASET COLUMNS
# ======================================
def normalize_dataset(df, text_col, label_col=None):
    df = df.copy()

    df = df.rename(
        columns={
            text_col: "text"
        }
    )

    if label_col:
        df = df.rename(
            columns={
                label_col: "label"
            }
        )

    return df

# ======================================
# LOAD LOCAL DATASETS
# ======================================
@st.cache_data
def load_local_datasets():

    # emotion_accuracy_training.csv
    emotion_train = pd.read_csv(
        "data/emotion_accuracy_training.csv"
    )

    emotion_train = normalize_dataset(
        emotion_train,
        "text",
        "label"
    )

    # ugm_fess_labeled.csv
    stress_df = pd.read_csv(
        "data/ugm_fess_labeled.csv"
    )

    stress_df = normalize_dataset(
        stress_df,
        "full_text",
        "label"
    )

    return emotion_train, stress_df

# ======================================
# LOAD HUGGINGFACE DATASETS
# ======================================
@st.cache_data
def load_hf_datasets():

    # indo slang
    indo_slang = load_dataset(
        "zeroix07/indo-slang-words"
    )

    indo_slang_df = pd.DataFrame(
        indo_slang["train"]
    )

    indo_slang_df = normalize_dataset(
        indo_slang_df,
        "text"
    )

    # english slang 1
    english_slang1 = load_dataset(
        "MariyaAnjum/genz-slang-dataset"
    )

    english_slang1_df = pd.DataFrame(
        english_slang1["train"]
    )

    english_slang1_df = normalize_dataset(
        english_slang1_df,
        "Slang"
    )

    # english slang 2
    english_slang2 = load_dataset(
        "acader/genz-alpha-slangs"
    )

    english_slang2_df = pd.DataFrame(
        english_slang2["train"]
    )

    english_slang2_df = normalize_dataset(
        english_slang2_df,
        "Sentence"
    )

    # go emotions
    english_emotion = load_dataset(
        "google-research-datasets/go_emotions"
    )

    english_emotion_df = pd.DataFrame(
        english_emotion["train"]
    )

    english_emotion_df = normalize_dataset(
        english_emotion_df,
        "text",
        "labels"
    )

    return (
        indo_slang_df,
        english_slang1_df,
        english_slang2_df,
        english_emotion_df
    )

emotion_train, stress_df = load_local_datasets()

(
    indo_slang,
    english_slang1,
    english_slang2,
    english_emotion
) = load_hf_datasets()

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:

    st.title("🧠 Stress Level App")

    pages = [
        "Home",
        "Dataset Overview",
        "Preprocessing",
        "Model Training",
        "Evaluation",
        "Prediction"
    ]

    for p in pages:
        if st.button(p):
            st.session_state.page = p
            st.rerun()

# ======================================
# HOME
# ======================================
if st.session_state.page == "Home":

    st.title("Stress Level Detection")

    st.write(
        "Deteksi stress level dari teks media sosial menggunakan NLP"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Stress Dataset",
        len(stress_df)
    )

    c2.metric(
        "Emotion Dataset",
        len(emotion_train)
    )

    c3.metric(
        "GoEmotion Dataset",
        len(english_emotion)
    )

# ======================================
# DATASET OVERVIEW
# ======================================
elif st.session_state.page == "Dataset Overview":

    st.title("Dataset Overview")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Stress",
        "Emotion Train",
        "Indo Slang",
        "English Slang",
        "GoEmotions"
    ])

    with tab1:
        st.dataframe(stress_df.head())

    with tab2:
        st.dataframe(emotion_train.head())

    with tab3:
        st.dataframe(indo_slang.head())

    with tab4:
        st.dataframe(english_slang1.head())

    with tab5:
        st.dataframe(english_emotion.head())

# ======================================
# PREPROCESSING
# ======================================
elif st.session_state.page == "Preprocessing":

    st.title("Text Preprocessing")

    stress_df["clean_text"] = stress_df["text"].apply(
        clean_text
    )

    st.success("Preprocessing selesai")

    st.dataframe(
        stress_df[
            ["text", "clean_text"]
        ].head(10)
    )

# ======================================
# MODEL TRAINING
# ======================================
elif st.session_state.page == "Model Training":

    st.title("Train Model")

    model_choice = st.selectbox(
        "Choose Model",
        [
            "Logistic Regression",
            "SVM"
        ]
    )

    stress_df["clean_text"] = stress_df["text"].apply(
        clean_text
    )

    X = stress_df["clean_text"]
    y = stress_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(
        X_train
    )

    X_test_vec = vectorizer.transform(
        X_test
    )

    if st.button("Train Model"):

        if model_choice == "Logistic Regression":
            model = LogisticRegression(
                max_iter=1000
            )
        else:
            model = SVC()

        model.fit(
            X_train_vec,
            y_train
        )

        y_pred = model.predict(
            X_test_vec
        )

        st.session_state.model = model
        st.session_state.vectorizer = vectorizer

        st.session_state.results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                average="weighted"
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                average="weighted"
            ),
            "f1": f1_score(
                y_test,
                y_pred,
                average="weighted"
            ),
            "y_test": y_test,
            "y_pred": y_pred
        }

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/stress_model.pkl"
        )

        joblib.dump(
            vectorizer,
            "models/tfidf.pkl"
        )

        st.success(
            "Model berhasil dilatih"
        )

# ======================================
# EVALUATION
# ======================================
elif st.session_state.page == "Evaluation":

    st.title("Model Evaluation")

    if st.session_state.results:

        result = st.session_state.results

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            round(result["accuracy"], 4)
        )

        c2.metric(
            "Precision",
            round(result["precision"], 4)
        )

        c3.metric(
            "Recall",
            round(result["recall"], 4)
        )

        c4.metric(
            "F1 Score",
            round(result["f1"], 4)
        )

        cm = confusion_matrix(
            result["y_test"],
            result["y_pred"]
        )

        fig, ax = plt.subplots()

        ax.imshow(cm)

        ax.set_title(
            "Confusion Matrix"
        )

        st.pyplot(fig)

    else:
        st.warning(
            "Train model dulu"
        )

# ======================================
# PREDICTION
# ======================================
elif st.session_state.page == "Prediction":

    st.title("Stress Prediction")

    user_text = st.text_area(
        "Masukkan teks sosial media"
    )

    if st.button("Predict"):

        if st.session_state.model is None:

            st.session_state.model = joblib.load(
                "models/stress_model.pkl"
            )

            st.session_state.vectorizer = joblib.load(
                "models/tfidf.pkl"
            )

        clean_input = clean_text(
            user_text
        )

        vectorized = st.session_state.vectorizer.transform(
            [clean_input]
        )

        prediction = st.session_state.model.predict(
            vectorized
        )[0]

        if prediction == 0:
            result = "Normal 😌"
        elif prediction == 1:
            result = "Mild Stress 😥"
        else:
            result = "High Stress 😫"

        st.success(
            f"Prediction: {result}"
        )

        st.subheader("Processed Text")
        st.write(clean_input)
