import streamlit as st
import pickle
import numpy as np
from helper import query_point_creator
import nltk
nltk.download('stopwords')


# Load saved models
rf_model = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
svd = pickle.load(open("svd.pkl", "rb"))

# UI
st.title("🔍 Quora Duplicate Question Detector")

q1 = st.text_input("Enter Question 1:")
q2 = st.text_input("Enter Question 2:")

if st.button("🔍Check Duplicate"):
    if not q1 or not q2:
        st.warning("Please enter both questions.")
    else:
        features = query_point_creator(q1, q2, vectorizer, svd)
        pred = rf_model.predict(features)[0]
        st.success("✅ Duplicate" if pred else "❌ Not Duplicate")

st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border: 2px solid #4CAF50;
        border-radius: 4px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)


# Footer (Bottom-right fixed)
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: gray;
        opacity: 0.8;
        z-index: 100;
    }
    </style>
    <div class="footer">
        Made by Priyanka Ranjan
    </div>
""", unsafe_allow_html=True)
