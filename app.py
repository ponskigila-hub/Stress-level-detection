import streamlit as st
import pandas as pd
import numpy as np
import os
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title='Stress Detection NLP App', layout='wide')

DATA_PATH_1 = 'data/emotion_accuracy_training.csv'
DATA_PATH_2 = 'data/ugm_fess_labeled.csv'

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df1 = pd.read_csv(DATA_PATH_1)
    df1 = df1.rename(columns={'tweet':'text'})

    df2 = pd.read_csv(DATA_PATH_2)
    # fix malformed label column
    label_col = [c for c in df2.columns if 'label' in c.lower()][0]
    df2 = df2.rename(columns={'full_text':'text', label_col:'stress_label'})
    df2['stress_label'] = df2['stress_label'].astype(str).str.replace(';', '', regex=False)
    df2 = df2[df2['stress_label'].isin(['0','1','2'])]
    df2['stress_label'] = df2['stress_label'].astype(int)

    return df1, df2

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Training
# -----------------------------
@st.cache_resource
def train_models(df_emotion, df_stress):
    df_emotion['clean_text'] = df_emotion['text'].apply(clean_text)
    df_stress['clean_text'] = df_stress['text'].apply(clean_text)

    # emotion model
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        df_emotion['clean_text'], df_emotion['label'], test_size=0.2, random_state=42
    )

    emotion_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC()
    }

    emotion_results = {}
    best_emotion_pipeline = None
    best_acc = 0

    for name, model in emotion_models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', model)
        ])
        pipeline.fit(X_train_e, y_train_e)
        preds = pipeline.predict(X_test_e)
        acc = accuracy_score(y_test_e, preds)
        emotion_results[name] = acc

        if acc > best_acc:
            best_acc = acc
            best_emotion_pipeline = pipeline

    # stress model (0=low,1=medium,2=high)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        df_stress['clean_text'], df_stress['stress_label'], test_size=0.2, random_state=42
    )

    stress_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    stress_pipeline.fit(X_train_s, y_train_s)
    stress_preds = stress_pipeline.predict(X_test_s)
    stress_acc = accuracy_score(y_test_s, stress_preds)

    return best_emotion_pipeline, stress_pipeline, emotion_results, stress_acc

# -----------------------------
# Main
# -----------------------------
df_emotion, df_stress = load_data()
emotion_model, stress_model, emotion_results, stress_acc = train_models(df_emotion, df_stress)

st.sidebar.title('NLP Pipeline Navigation')
menu = st.sidebar.radio(
    'Choose Section',
    ['EDA', 'Text Preprocessing', 'Feature Engineering', 'Model Selection & Evaluation', 'Prediction Demo']
)

if menu == 'EDA':
    st.title('Exploratory Data Analysis')

    st.subheader('Emotion Dataset Sample')
    st.dataframe(df_emotion.head())

    st.subheader('Emotion Distribution')
    fig, ax = plt.subplots()
    df_emotion['label'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader('Stress Dataset Distribution')
    fig, ax = plt.subplots()
    df_stress['stress_label'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    all_text = ' '.join(df_emotion['text'].astype(str).tolist())
    wc = WordCloud(width=800, height=400).generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')
    st.pyplot(fig)

elif menu == 'Text Preprocessing':
    st.title('Text Preprocessing')
    sample = df_emotion[['text']].head(10).copy()
    sample['cleaned'] = sample['text'].apply(clean_text)
    st.dataframe(sample)

elif menu == 'Feature Engineering':
    st.title('TF-IDF Feature Engineering')
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=20)
    X = vectorizer.fit_transform(df_emotion['text'].apply(clean_text))

    feature_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    st.dataframe(feature_df.head())

elif menu == 'Model Selection & Evaluation':
    st.title('Model Selection & Evaluation')

    st.subheader('Emotion Model Accuracy Comparison')
    result_df = pd.DataFrame(
        emotion_results.items(), columns=['Model', 'Accuracy']
    )
    st.dataframe(result_df)

    fig, ax = plt.subplots()
    sns.barplot(data=result_df, x='Model', y='Accuracy', ax=ax)
    st.pyplot(fig)

    st.metric('Stress Model Accuracy', f'{stress_acc:.2%}')

elif menu == 'Prediction Demo':
    st.title('Stress & Emotion Prediction Demo')

    user_input = st.text_area('Input your text here')

    if st.button('Predict'):
        cleaned = clean_text(user_input)

        emotion_pred = emotion_model.predict([cleaned])[0]
        stress_pred = stress_model.predict([cleaned])[0]
        stress_proba = stress_model.predict_proba([cleaned])[0]

        stress_percentage = int(np.max(stress_proba) * 100)

        stress_map = {
            0: 'Low Stress',
            1: 'Medium Stress',
            2: 'High Stress'
        }

        st.success(f'Predicted Emotion: {emotion_pred}')
        st.warning(f'Stress Level: {stress_map[stress_pred]}')
        st.info(f'Stress Percentage: {stress_percentage}%')

        fig, ax = plt.subplots()
        ax.bar(['Low', 'Medium', 'High'], stress_proba)
        ax.set_ylabel('Probability')
        st.pyplot(fig)
