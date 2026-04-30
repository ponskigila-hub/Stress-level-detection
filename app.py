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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title='Interactive Stress Detection NLP App', layout='wide')

# Load data
@st.cache_data
def load_data():
    df_emotion = pd.read_csv('data/emotion_accuracy_training.csv')
    df_emotion = df_emotion.rename(columns={'tweet': 'text'})

    df_stress = pd.read_csv('data/ugm_fess_labeled.csv')
    label_col = [c for c in df_stress.columns if 'label' in c.lower()][0]
    df_stress = df_stress.rename(columns={'full_text': 'text', label_col: 'stress_label'})
    df_stress['stress_label'] = df_stress['stress_label'].astype(str).str.replace(';','')
    df_stress['stress_label'] = pd.to_numeric(df_stress['stress_label'], errors='coerce')
    df_stress = df_stress.dropna()

    return df_emotion, df_stress


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()


def build_pipeline(model_name):
    model_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(),
        'Random Forest': RandomForestClassifier()
    }

    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ('clf', model_dict[model_name])
    ])

# Sidebar controls
st.sidebar.title('NLP Pipeline')
menu = st.sidebar.radio('Navigation', [
    'EDA',
    'Text Preprocessing',
    'Feature Engineering',
    'Model Selection & Evaluation',
    'Prediction Demo'
])

train_ratio = st.sidebar.slider('Training Set Ratio', 0.6, 0.9, 0.8, 0.05)
model_choice = st.sidebar.selectbox(
    'Choose Model',
    ['Logistic Regression', 'Naive Bayes', 'Linear SVM', 'Random Forest']
)

# Load datasets
df_emotion, df_stress = load_data()
df_emotion['clean_text'] = df_emotion['text'].apply(clean_text)
df_stress['clean_text'] = df_stress['text'].apply(clean_text)

# Train model dynamically
pipeline = build_pipeline(model_choice)
X_train, X_test, y_train, y_test = train_test_split(
    df_emotion['clean_text'],
    df_emotion['label'],
    train_size=train_ratio,
    random_state=42
)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)

# Separate stress model for percentage
stress_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])
stress_pipeline.fit(df_stress['clean_text'], df_stress['stress_label'])

# EDA Section
if menu == 'EDA':
    st.title('Exploratory Data Analysis')

    eda_option = st.selectbox('Select EDA Visualization', [
        'Dataset Preview',
        'Emotion Distribution',
        'Stress Distribution',
        'Text Length Distribution',
        'WordCloud'
    ])

    if eda_option == 'Dataset Preview':
        st.dataframe(df_emotion.head(20))

    elif eda_option == 'Emotion Distribution':
        fig, ax = plt.subplots()
        df_emotion['label'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    elif eda_option == 'Stress Distribution':
        fig, ax = plt.subplots()
        df_stress['stress_label'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    elif eda_option == 'Text Length Distribution':
        df_emotion['length'] = df_emotion['text'].apply(len)
        fig, ax = plt.subplots()
        sns.histplot(df_emotion['length'], kde=True, ax=ax)
        st.pyplot(fig)

    elif eda_option == 'WordCloud':
        text = ' '.join(df_emotion['text'])
        wc = WordCloud(width=1000, height=500).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')
        st.pyplot(fig)

# Preprocessing
elif menu == 'Text Preprocessing':
    st.title('Text Preprocessing')
    sample_size = st.slider('Sample Rows', 5, 50, 10)
    sample = df_emotion[['text']].sample(sample_size)
    sample['cleaned'] = sample['text'].apply(clean_text)
    st.dataframe(sample)

# Feature Engineering
elif menu == 'Feature Engineering':
    st.title('Feature Engineering (TF-IDF)')
    tfidf = pipeline.named_steps['tfidf']
    features = tfidf.fit_transform(df_emotion['clean_text'])
    feature_names = tfidf.get_feature_names_out()[:30]

    st.write('Top 30 Features')
    st.write(feature_names)

# Model Evaluation
elif menu == 'Model Selection & Evaluation':
    st.title('Model Evaluation')

    st.metric('Accuracy', f'{acc:.2%}')

    st.subheader('Classification Report')
    report = classification_report(y_test, preds, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

# Prediction Demo
elif menu == 'Prediction Demo':
    st.title('Prediction Demo')

    user_input = st.text_area('Input text for prediction')

    if st.button('Predict'):
        cleaned = clean_text(user_input)

        emotion_pred = pipeline.predict([cleaned])[0]

        # Probability handling for better confidence
        if hasattr(stress_pipeline.named_steps['clf'], 'predict_proba'):
            stress_proba = stress_pipeline.predict_proba([cleaned])[0]
            stress_percentage = round(max(stress_proba) * 100, 2)
            stress_level = np.argmax(stress_proba)
        else:
            stress_level = stress_pipeline.predict([cleaned])[0]
            stress_percentage = 75

        stress_map = {
            0: 'Low Stress',
            1: 'Medium Stress',
            2: 'High Stress'
        }

        st.success(f'Emotion: {emotion_pred}')
        st.warning(f'Stress Level: {stress_map[stress_level]}')
        st.info(f'Stress Percentage: {stress_percentage}%')

        fig, ax = plt.subplots()
        if 'stress_proba' in locals():
            ax.bar(['Low', 'Medium', 'High'], stress_proba)
        st.pyplot(fig)
