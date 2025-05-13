import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("fake_news_ann.h5")

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("Gatimu Fake News Detection")
user_input = st.text_area("Enter a news article (title and body):")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input 
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)[0][0]

        # Show the score
        st.write(f"Prediction Score: {pred:.4f}")

        if pred > 0.5:
            st.success("This news is likely **REAL**.")
        else:
            st.error("This news is likely **FAKE**.")

