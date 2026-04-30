import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import joblib

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
# SESSION STATE
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "model" not in st.session_state:
    st.session_state.model = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "results" not in st.session_state:
    st.session_state.results = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

stemmer = PorterStemmer()

# ======================================
# CLEAN TEXT
# ======================================
def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# ======================================
# LOAD DATASETS
# ======================================
@st.cache_data
def load_datasets():

    # Stress dataset
    stress_df = pd.read_csv(
        "data/ugm_fess_labeled.csv",
        sep=";"
    )

    stress_df.columns = (
        stress_df.columns
        .str.strip()
        .str.replace(";", "", regex=False)
    )

    if "full_text" in stress_df.columns:
        stress_df = stress_df.rename(
            columns={
                "full_text": "text"
            }
        )

    # Emotion dataset
    emotion_df = pd.read_csv(
        "data/emotion_accuracy_training.csv"
    )

    emotion_df.columns = (
        emotion_df.columns
        .str.strip()
    )

    return stress_df, emotion_df

stress_df, emotion_df = load_datasets()

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
        "Deteksi tingkat stress pengguna media sosial menggunakan NLP"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Stress Dataset",
        len(stress_df)
    )

    c2.metric(
        "Emotion Dataset",
        len(emotion_df)
    )

# ======================================
# DATASET OVERVIEW
# ======================================
elif st.session_state.page == "Dataset Overview":

    st.title("Dataset Overview")

    tab1, tab2 = st.tabs([
        "Stress Dataset",
        "Emotion Dataset"
    ])

    with tab1:
        st.subheader("UGM Stress Dataset")
        st.dataframe(stress_df.head())

    with tab2:
        st.subheader("Emotion Dataset")
        st.dataframe(emotion_df.head())

# ======================================
# PREPROCESSING
# ======================================
elif st.session_state.page == "Preprocessing":

    st.title("🧹 Text Preprocessing")

    if "text" not in stress_df.columns:
        st.error("Kolom text tidak ditemukan")
        st.stop()

    if "label" not in stress_df.columns:
        st.error("Kolom label tidak ditemukan")
        st.stop()

    st.subheader("Before Preprocessing")

    st.dataframe(
        stress_df.head(10)
    )

    if st.button("Run Preprocessing"):

        processed_df = stress_df.copy()

        processed_df = processed_df[
            processed_df["text"].notna()
        ]

        processed_df["clean_text"] = processed_df[
            "text"
        ].apply(clean_text)

        st.session_state.processed_df = processed_df

        st.success(
            "Preprocessing selesai"
        )

        preview_df = pd.DataFrame({
            "Original": processed_df["text"].head(10),
            "Processed": processed_df["clean_text"].head(10)
        })

        st.dataframe(preview_df)

# ======================================
# MODEL TRAINING
# ======================================
elif st.session_state.page == "Model Training":

    st.title("Train Model")

    if st.session_state.processed_df is None:
        st.warning(
            "Lakukan preprocessing terlebih dahulu"
        )
        st.stop()

    model_choice = st.selectbox(
        "Choose Model",
        [
            "Logistic Regression",
            "SVM"
        ]
    )

    processed_df = st.session_state.processed_df

    X = processed_df["clean_text"]
    y = processed_df["label"]

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
            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),
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
            "Train model terlebih dahulu"
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

            if os.path.exists(
                "models/stress_model.pkl"
            ):
                st.session_state.model = joblib.load(
                    "models/stress_model.pkl"
                )

                st.session_state.vectorizer = joblib.load(
                    "models/tfidf.pkl"
                )
            else:
                st.error(
                    "Model belum ada. Train model dulu."
                )
                st.stop()

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
