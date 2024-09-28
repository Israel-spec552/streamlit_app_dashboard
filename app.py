import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = tf.keras.models.load_model('model.h5')
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')  # Make sure this matches your training tokenizer
max_length = 100  # Ensure this is the same as used during training

# Preprocessing function (for example)
def preprocess_data(data):
    # Example preprocessing logic
    # This should be customized based on your dataset and model requirements
    return data

# Streamlit app code
def run_app():
    st.title('Sentiment Prediction Model')

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    features = None

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.write("Data Loaded Successfully:")
        st.write(data)

        # Preprocess the data
        features = preprocess_data(data)

        if features is None:
            st.write("Please upload a CSV file with numeric data.")

    # Text input for manual sentiment analysis
    user_input = st.text_area("Enter text for sentiment analysis:")

    # Make predictions
    if st.button('Predict'):
        # Handle CSV predictions
        if features is not None:
            features = features.astype(np.float32)
            predictions = model.predict(features)
            predicted_classes = np.argmax(predictions, axis=1)

            sentiment_map = {0: 'Negative', 1: 'Positive'}
            predicted_sentiments = [sentiment_map[pred] for pred in predicted_classes]
            st.write(f'Predictions from uploaded CSV: {predicted_sentiments}')
        
        # Handle text input predictions
        if user_input:
            # Tokenize and pad the input text
            sequences = tokenizer.texts_to_sequences([user_input])  # Convert text to sequence
            padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

            # Predict sentiment
            user_prediction = model.predict(padded_sequences)
            predicted_class = np.argmax(user_prediction, axis=1)

            # Map prediction to sentiment
            sentiment_map = {0: 'Negative', 1: 'Positive'}
            sentiment = sentiment_map.get(predicted_class[0], 'Unknown')
            probability = user_prediction[0][predicted_class[0]]

            # Display the sentiment and probability
            st.write(f"Sentiment for entered text: {sentiment} (Probability: {probability:.2f})")
            
            # If sentiment seems incorrect, debug the probabilities
            st.write(f"Model's raw output: {user_prediction}")

        else:
            st.write("Please upload a CSV file or enter text for sentiment analysis.")

# Run the app
run_app()
