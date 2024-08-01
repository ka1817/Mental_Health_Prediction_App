
import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

 #Load stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return ''

def main():
    st.title('Mental Health Prediction App')

    user_input = st.text_area("Enter your Mental Condition:")

    if st.button("Predict"):

        preprocessed_input = preprocess_text(user_input)

        text_tfidf = vectorizer.transform([preprocessed_input])

        prediction = model.predict(text_tfidf)
        st.write(f"Prediction : {prediction[0]}")
    else:
        print("Enter your mental condition")
if __name__ == "__main__":
    main()
