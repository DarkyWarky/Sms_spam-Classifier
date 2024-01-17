import streamlit as st
import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
response = requests.get(url)
with ZipFile(BytesIO(response.content)) as z:
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', names=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
df['message'] = df['message'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word.lower() not in stop_words]))

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X_vectorized, y)

def predict_spam(message):
    processed_message = ' '.join([ps.stem(word) for word in message.split() if word.lower() not in stop_words])
    input_vectorized = vectorizer.transform([processed_message])
    prediction = model.predict(input_vectorized)[0]
    return prediction

st.title("SMS Classifier")

message = st.text_input("Enter SMS Message:")
if message:
    prediction = predict_spam(message)
    st.write(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
