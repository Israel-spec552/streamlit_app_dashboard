import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

# Ensure the tokenizer is fitted (this should ideally be done during model training)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
# Assuming you have the word index from training, load it like below:
# tokenizer.word_index = {'some_word': 1, 'another_word': 2, ...}

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
        if features is not None:
            features = features.astype(np.float32)
            predictions = model.predict(features)
            predicted_classes = np.argmax(predictions, axis=1)
            sentiment_map = {0: 'Negative', 1: 'Positive'}
            predicted_sentiments = [sentiment_map[pred] for pred in predicted_classes]
            st.write(f'Predictions from uploaded CSV: {predicted_sentiments}')

        if user_input:
            # Preprocess the text input
            sequences = tokenizer.texts_to_sequences([user_input])  # Convert text to sequence
            
            # Handle empty or invalid sequences
            if not sequences or all(len(seq) == 0 for seq in sequences):
                st.write("Input text could not be processed by the tokenizer.")
                return
            
            # Pad the sequences
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust `maxlen` as per model training

            # Predict sentiment for user-entered text
            user_prediction = model.predict(padded_sequences)
            predicted_class = np.argmax(user_prediction, axis=1)
            confidence = user_prediction[0][predicted_class[0]]

            # Set a threshold for prediction confidence (e.g., 60%)
            threshold = 0.6
            sentiment_map = {0: 'Negative', 1: 'Positive'}

            if confidence >= threshold:
                sentiment = sentiment_map.get(predicted_class[0], 'Unknown')
            else:
                sentiment = 'Uncertain'

            st.write(f"Sentiment for entered text: {sentiment} (Probability: {confidence:.2f})")
        else:
            st.write("Please upload a CSV file or enter text for sentiment analysis.")

# Run the app
run_app()
