import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import RandomOverSampler


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Stress Detection App",
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1f2937);
}

[data-testid="stSidebar"] * {
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #475569;
}

.metric-title {
    font-size: 15px;
    color: #cbd5e1;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}

.big-card {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    padding: 25px;
    border-radius: 20px;
    color: white;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    font-weight: bold;
}

h1, h2, h3 {
    color: #60a5fa;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================
if "emotion_model" not in st.session_state:
    st.session_state.emotion_model = None

if "stress_model" not in st.session_state:
    st.session_state.stress_model = None


# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    emotion_df = pd.read_csv(
        "data/emotion_accuracy_training.csv"
    )

    emotion_df = emotion_df.rename(
        columns={
            "tweet": "text"
        }
    )

    stress_df = pd.read_csv(
        "data/ugm_fess_labeled.csv"
    )

    label_col = [
        c for c in stress_df.columns
        if "label" in c.lower()
    ][0]

    stress_df = stress_df.rename(
        columns={
            "full_text": "text",
            label_col: "stress_label"
        }
    )

    stress_df["stress_label"] = (
        stress_df["stress_label"]
        .astype(str)
        .str.replace(";", "")
    )

    stress_df["stress_label"] = pd.to_numeric(
        stress_df["stress_label"],
        errors="coerce"
    )

    stress_df = stress_df.dropna()

    return emotion_df, stress_df


# ==========================================
# TEXT CLEANING
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


# ==========================================
# BUILD MODEL
# ==========================================
def build_model(name):

    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    elif name == "Naive Bayes":
        return MultinomialNB()

    elif name == "Linear SVM":
        return LinearSVC(
            class_weight="balanced"
        )


# ==========================================
# LOAD DATASET
# ==========================================
emotion_df, stress_df = load_data()

emotion_df["clean_text"] = emotion_df["text"].apply(
    clean_text
)

stress_df["clean_text"] = stress_df["text"].apply(
    clean_text
)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.title("🧠 Stress Detection")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "EDA",
            "Preprocessing",
            "Model",
            "Prediction"
        ]
    )


# ==========================================
# HOME
# ==========================================
if page == "Home":

    st.title("🧠 Stress Detection NLP App")

    st.markdown("""
    <div class="big-card">
        Detect emotion and stress level from social media text
        using Natural Language Processing and Machine Learning.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Emotion Dataset</div>
            <div class="metric-value">{len(emotion_df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Stress Dataset</div>
            <div class="metric-value">{len(stress_df)}</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# EDA
# ==========================================
elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Emotion Dataset",
        "Stress Dataset",
        "WordCloud"
    ])

    with tab1:
        st.dataframe(
            emotion_df.head()
        )

        fig, ax = plt.subplots()

        emotion_df["label"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Emotion Distribution"
        )

        st.pyplot(fig)

    with tab2:
        st.dataframe(
            stress_df.head()
        )

        fig, ax = plt.subplots()

        stress_df["stress_label"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Stress Distribution"
        )

        st.pyplot(fig)

    with tab3:
        text = " ".join(
            emotion_df["text"]
        )

        wc = WordCloud(
            width=1200,
            height=500
        ).generate(text)

        fig, ax = plt.subplots(
            figsize=(15,5)
        )

        ax.imshow(wc)
        ax.axis("off")

        st.pyplot(fig)


# ==========================================
# PREPROCESSING
# ==========================================
elif page == "Preprocessing":

    st.title("⚙ Text Preprocessing")

    sample = emotion_df.sample(
        10
    ).copy()

    preview_df = pd.DataFrame({
        "Original Text": sample["text"],
        "Cleaned Text": sample["clean_text"]
    })

    st.dataframe(
        preview_df,
        use_container_width=True
    )


# ==========================================
# MODEL
# ==========================================
elif page == "Model":

    st.title("🤖 Model Training")

    model_name = st.selectbox(
        "Choose Model",
        [
            "Logistic Regression",
            "Naive Bayes",
            "Linear SVM"
        ]
    )

    if st.button("Train Model"):

        # Emotion
        X_emotion = emotion_df["clean_text"]
        y_emotion = emotion_df["label"]

        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
            X_emotion,
            y_emotion,
            test_size=0.2,
            random_state=42
        )

        emotion_pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000
                )
            ),
            (
                "clf",
                build_model(model_name)
            )
        ])

        emotion_pipeline.fit(
            X_train_e,
            y_train_e
        )

        pred_emotion = emotion_pipeline.predict(
            X_test_e
        )

        # Stress
        X_stress = stress_df["clean_text"]
        y_stress = stress_df["stress_label"]

        tfidf = TfidfVectorizer(
            max_features=10000
        )

        X_vec = tfidf.fit_transform(
            X_stress
        )

        ros = RandomOverSampler(
            random_state=42
        )

        X_resampled, y_resampled = ros.fit_resample(
            X_vec,
            y_stress
        )

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_resampled,
            y_resampled,
            test_size=0.2,
            random_state=42
        )

        stress_model = build_model(
            model_name
        )

        stress_model.fit(
            X_train_s,
            y_train_s
        )

        pred_stress = stress_model.predict(
            X_test_s
        )

        st.session_state.emotion_model = emotion_pipeline
        st.session_state.stress_model = (
            stress_model,
            tfidf
        )

        st.success(
            "Training Completed"
        )

        # Metrics
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            round(
                accuracy_score(
                    y_test_s,
                    pred_stress
                ),
                3
            )
        )

        c2.metric(
            "Precision",
            round(
                precision_score(
                    y_test_s,
                    pred_stress,
                    average="weighted"
                ),
                3
            )
        )

        c3.metric(
            "Recall",
            round(
                recall_score(
                    y_test_s,
                    pred_stress,
                    average="weighted"
                ),
                3
            )
        )

        c4.metric(
            "F1 Score",
            round(
                f1_score(
                    y_test_s,
                    pred_stress,
                    average="weighted"
                ),
                3
            )
        )

        st.subheader(
            "Confusion Matrix"
        )

        fig, ax = plt.subplots(
            figsize=(8,6)
        )

        cm = confusion_matrix(
            y_test_s,
            pred_stress
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        st.pyplot(fig)

        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y_test_s,
            pred_stress,
            output_dict=True
        )

        st.dataframe(
            pd.DataFrame(report).transpose()
        )


# ==========================================
# PREDICTION
# ==========================================
elif page == "Prediction":

    st.title("🔮 Stress Prediction")

    user_input = st.text_area(
        "Input text here"
    )

    if st.button("Predict"):

        if (
            st.session_state.emotion_model is None
            or
            st.session_state.stress_model is None
        ):
            st.warning(
                "Train model first"
            )

        else:
            cleaned = clean_text(
                user_input
            )

            emotion_pred = (
                st.session_state.emotion_model
                .predict([cleaned])[0]
            )

            stress_model, tfidf = (
                st.session_state.stress_model
            )

            stress_vec = tfidf.transform(
                [cleaned]
            )

            stress_pred = stress_model.predict(
                stress_vec
            )[0]

            stress_map = {
                0: "Normal 😌",
                1: "Mild Stress 😥",
                2: "High Stress 😫"
            }

            c1, c2 = st.columns(2)

            with c1:
                st.success(
                    f"Emotion: {emotion_pred}"
                )

            with c2:
                st.warning(
                    f"Stress: {stress_map[stress_pred]}"
                )
