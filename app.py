import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

model_path = 'model.h5'  # Ensure this is the correct relative or absolute path
model = tf.keras.models.load_model(model_path)

# Function to preprocess data
def preprocess_data(data):
    # Convert categorical data to numerical using LabelEncoder
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])

    # Check if there are any numeric columns left after conversion
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        st.write("No numeric data found in the uploaded file after preprocessing.")
        return None

    # Normalize the numeric data
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(numeric_data)

    return numeric_data

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
            # Preprocess the text input (using tokenizer and padding)
            sequences = tokenizer.texts_to_sequences([user_input])  # Convert text to sequence
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per model training

            # Predict sentiment for user-entered text
            user_prediction = model.predict(padded_sequences)
            predicted_class = np.argmax(user_prediction, axis=1)

            sentiment_map = {0: 'Negative', 1: 'Positive'}
            sentiment = sentiment_map.get(predicted_class[0], 'Unknown')
            st.write(f"Sentiment for entered text: {sentiment} (Probability: {user_prediction[0][predicted_class[0]]:.2f})")
        else:
            st.write("Please upload a CSV file or enter text for sentiment analysis.")

# Run the app
run_app()                          
