import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Load the pre-trained model
model_path = 'model.h5'  # Ensure this is the correct relative or absolute path
model = tf.keras.models.load_model(model_path)


# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to predict sentiment for each review individually
def predict_individual_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    sentiment = sentiment_scores['compound']

    # Determine if the sentiment is positive, negative, or neutral
    if sentiment >= 0.05:
        sentiment_label = "Positive"
    elif sentiment <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Convert compound score to percentage (between 0 to 100)
    sentiment_percentage = abs(sentiment) * 100

    return f"{sentiment_label} ({sentiment_percentage:.2f}%)"

# Set up the Streamlit app layout
st.title("Sentiment Analyzer Web App")
st.write("This app allows you to analyze the sentiment of individual text reviews or a CSV dataset containing text reviews.")

# Upload CSV file section
st.subheader("Upload CSV Dataset (Optional)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type="csv")

# If a CSV file is uploaded
if uploaded_file is not None:
    try:
        # Load the CSV file into a dataframe
        df = pd.read_csv(uploaded_file)

        # Check if the 'review' column exists
        if 'review' in df.columns:
            st.write(f"Loaded {len(df)} reviews from the CSV file.")
            
            # Analyze sentiment of the dataset
            df['Sentiment'] = df['review'].apply(lambda review: predict_individual_sentiment(review))

            # Show the dataframe with sentiment analysis
            st.write("Sentiment analysis results for the dataset:")
            st.dataframe(df[['review', 'Sentiment']])
        else:
            st.error("The CSV file must have a 'review' column.")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Text input for individual sentiment prediction
st.subheader("Enter Text for Sentiment Analysis")
user_text = st.text_area("Enter a sentence or paragraph below:")

# Button to trigger sentiment analysis on the entered text
if st.button("Predict Sentiment"):
    if user_text:
        # Get the sentiment of the entered text
        result = predict_individual_sentiment(user_text)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter some text for sentiment analysis.")
run_app()
