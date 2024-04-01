import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import load, dump  # Use joblib to save and load models
  # Use joblib to save and load models
from gensim.models import Word2Vec
import nltk
nltk.download('stopwords')


# Function to preprocess text
def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

# Load the Naive Bayes model
classifier = joblib.load('naive_bayes_model.pkl')

# Load TF-IDF vectorizer
tfidf_v = joblib.load('tfidf_vectorizer.pkl')

# Load Word2Vec model
word2vec_model=Word2Vec()
# word2vec_model = Word2Vec.load('word2vec_model.bin')

# Function to generate document embeddings using Word2Vec
def document_embedding(review, model):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Function to predict fake/real news
def predict_news(text):
    processed_text = preprocess_text(text)
    val = tfidf_v.transform([processed_text]).toarray()
    embedding = document_embedding(processed_text, word2vec_model)
    val = np.hstack((val, embedding.reshape(1, -1)))
    prediction = classifier.predict(val)
    return prediction[0]

# Streamlit app
def main():
    st.title("Fake News Detection")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Home", "Detect News"])

    if page == "Home":
        st.header("Welcome to Fake News Detection App")
        st.write("This app detects whether a given news article is fake or real.")

    elif page == "Detect News":
        st.header("Detect Fake or Real News")
        news_text = st.text_area("Enter the news text:")
        if st.button("Detect"):
            if news_text.strip() != "":
                prediction = predict_news(news_text)
                if prediction == 0:
                    st.write("Prediction: Real News")
                elif prediction ==1:
                    st.write("Prediction: Fake News")
            else:
                st.write("Please enter some text to detect.")

if __name__ == "__main__":
    main()
