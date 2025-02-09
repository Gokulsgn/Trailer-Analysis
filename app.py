import streamlit as st
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.linear_model import LinearRegression

# Mock model and data (Replace with actual trained model)
model = LinearRegression()
model.coef_ = np.array([1000, 50000, 20000])  # Mock coefficients
model.intercept_ = 1000000  # Mock intercept

# Function to calculate sentiment score
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

# Custom CSS for a modern look
st.markdown("""
    <style>
    body {
        background-color: #F8F9FA;
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        margin: 40px;
    }
    h1 {
        color: #333333;
        text-align: center;
        font-weight: 700;
    }
    .stButton button {
        background-color: #007BFF;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .slider-container {
        text-align: center;
    }
    .stTextArea textarea {
        height: 200px;
        font-size: 14px;
    }
    .predict-button {
        margin-top: 20px;
        text-align: center;
    }
    .output {
        text-align: center;
        font-size: 24px;
        color: #28a745;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit App layout
st.title('🎬 Movie Success Predictor')

st.write("""
    This app predicts the estimated box office revenue based on trailer length, tone, and text analysis.
    Fill in the details below to get your prediction.
""")

# Inputs
st.markdown("<h3>🎥 Trailer Details</h3>", unsafe_allow_html=True)

trailer_length = st.slider('⏳ Trailer Length (seconds)', min_value=30, max_value=300, value=120, step=10)
trailer_tone = st.selectbox('📽️ Trailer Tone', ['positive', 'neutral', 'negative'])
trailer_text = st.text_area('✍️ Trailer Text (Enter a brief description of the trailer)')

# Convert tone to numerical
tone_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
trailer_tone_num = tone_mapping[trailer_tone]

# Sentiment score
sentiment_score = get_sentiment_score(trailer_text)

# Predict Button
st.markdown("<div class='predict-button'>", unsafe_allow_html=True)
if st.button('🔮 Predict Box Office Revenue'):
    input_features = np.array([[trailer_length, trailer_tone_num, sentiment_score]])
    predicted_revenue = model.predict(input_features)[0]
    st.markdown(f"<p class='output'>🎉 Predicted Box Office Revenue: ${predicted_revenue:,.2f}</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
