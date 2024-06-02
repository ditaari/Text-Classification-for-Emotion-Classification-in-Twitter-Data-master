import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import altair as alt

# Fungsi untuk memuat dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File tidak ditemukan di {file_path}. Silakan periksa jalur file.")
        return None

# Mendefinisikan jalur ke dataset
file_path = 'Twitter_Emotion_Dataset.csv'

# Memuat dataset
df = load_data(file_path)

# Lanjutkan jika dataset berhasil dimuat
if df is not None:
    # Encode label
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])

    # Mendefinisikan fungsi pembersihan teks
    def cleaning(x):
        x = x.strip()
        x = x.lower()
        x = re.sub(r'\d+', '', x)
        x = x.translate(str.maketrans('', '', string.punctuation))
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        x = stopword.remove(x)
        return x

    # Menerapkan fungsi pembersihan ke tweet
    df['tweet'] = df['tweet'].apply(lambda x: cleaning(x))

    # Membagi data menjadi set pelatihan dan pengujian
    X = df['tweet']
    y = df['label_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    # Memvektorisasi data teks
    count_vec = CountVectorizer()
    X_train_c = count_vec.fit_transform(X_train)
    X_test_c = count_vec.transform(X_test)

    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train_c)
    X_test_tfidf = tfidf.transform(X_test_c)

    # Melatih model
    mnb_model = MultinomialNB().fit(X_train_tfidf, y_train)

    # Menyimpan model dan vektorizer menggunakan joblib
    joblib.dump(mnb_model, 'mnb_model.pkl')
    joblib.dump(count_vec, 'count_vec.pkl')
    joblib.dump(tfidf, 'tfidf.pkl')

    # Memuat model dan vektorizer untuk prediksi
    mnb_model = joblib.load('mnb_model.pkl')
    count_vec = joblib.load('count_vec.pkl')
    tfidf = joblib.load('tfidf.pkl')

    # Streamlit application
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("twitter.png", width=100)
    with col2:
        st.title("Klasifikasi Emosi Tweet di Twitter")

    user_input = st.text_area("Masukkan teks tweet yang akan diklasifikasikan:")

    if st.button("Klasifikasikan"):
        if user_input:
            # Preprocess input pengguna
            clean_text = cleaning(user_input)
            clean_text_vec = count_vec.transform([clean_text])
            clean_text_tfidf = tfidf.transform(clean_text_vec)
            
            # Melakukan prediksi
            prediction = mnb_model.predict(clean_text_tfidf)
            prediction_proba = mnb_model.predict_proba(clean_text_tfidf)[0]
            
            # Memetakan label ke emosi dengan emotikon
            emotion_dict = {'anger': 'Marah 😡', 'happy': 'Bahagia 😍', 'sadness': 'Sedih 😔', 'fear': 'Ketakutan 😨', 'love': 'Cinta ❤️'}
            labels = le.classes_
            
            # Memastikan semua label ada dalam emotion_dict
            for label in labels:
                if label not in emotion_dict:
                    st.error(f"Label '{label}' tidak ditemukan dalam emotion_dict.")
                    emotion_dict[label] = label

            label_with_emojis = [emotion_dict[label] for label in labels]
            
            # Menampilkan hasil
            st.write("Prediksi Emosi:", emotion_dict[labels[prediction[0]]])
            st.write("Probabilitas Prediksi:", prediction_proba)
            
            # Menampilkan probabilitas prediksi dalam grafik batang
            df_proba = pd.DataFrame({'Emosi': label_with_emojis, 'Probabilitas': prediction_proba})
            fig = alt.Chart(df_proba).mark_bar().encode(
                x='Emosi',
                y='Probabilitas',
                color='Emosi'
            ).properties(
                width=alt.Step(80)
            )
            st.altair_chart(fig, use_container_width=True)
        else:
            st.write("Silakan masukkan teks untuk diklasifikasikan.")
