import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the pre-trained model
model_path = 'model.h5'  # Ensure this is the correct relative or absolute path
model = tf.keras.models.load_model(model_path)

# Load the original tokenizer used during training
# Ensure that the tokenizer was saved as a pickle file during the training
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess CSV data
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

# Function to preprocess user input text
def preprocess_user_input(text):
    # Tokenize and pad the text to the same length as used in training
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen according to your model's training setup
    return padded_sequences

# Streamlit app code
def run_app():
    st.title('TensorFlow/Keras Sentiment Prediction')

    # Initialize features to None
    features = None

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        # Load the CSV using pandas
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.write("Data Loaded Successfully:")
        st.write(data)

        # Preprocess the data (convert categorical data to numerical)
        features = preprocess_data(data)

        if features is None:
            st.write("Please upload a CSV file with numeric data.")

    # Text input box for manual sentiment analysis
    user_input = st.text_area("Enter text for sentiment analysis:")

    # Make predictions
    if st.button('Predict'):
        if user_input:
            # Preprocess user input text
            input_data = preprocess_user_input(user_input)

            # Convert to float32 (this assumes your model requires numeric input)
            input_data = input_data.astype(np.float32)

            # Predict sentiment for user-entered text
            user_prediction = model.predict(input_data)
            predicted_class = np.argmax(user_prediction, axis=1)

            # Print prediction probabilities for debugging
            st.write(f"Prediction probabilities: {user_prediction}")

            # Map prediction to 'Negative' or 'Positive'
            sentiment_map = {0: 'Negative', 1: 'Positive'}
            sentiment = sentiment_map.get(predicted_class[0], 'Unknown')

            st.write(f"Sentiment for entered text: {sentiment} (Probability: {user_prediction[0][predicted_class[0]]:.2f})")
        else:
            st.write("Please upload a CSV file or enter text for sentiment analysis.")

# Run the app
run_app()
