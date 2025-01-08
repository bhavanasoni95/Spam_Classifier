import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Ensure punkt and stopwords are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

def transform_text(text):
    # 1. Lower Case
    text = text.lower()

    # 2. Tokenization
    text = word_tokenize(text)

    # 3. Removing Special characters
    text = [re.sub(r'[@#$]', '', t) for t in text]

    # 4. Removing Stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]

    # 5. Stemming
    st = PorterStemmer()
    text = [st.stem(word) for word in text]

    # 6. Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    return " ".join(text)

# Streamlit UI
st.title("Spam Classifier")
input_sms = st.text_input("Enter your message:")

if input_sms:
    # 1) Preprocess the input text
    transform_sms = transform_text(input_sms)

    # 2) Vectorize using the loaded TF-IDF vectorizer
    vector_input = tfidf.transform([transform_sms])

    # 3) Predict using the loaded model
    result = model.predict(vector_input)[0]

    # 4) Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
