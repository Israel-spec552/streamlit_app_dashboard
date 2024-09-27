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
    st.title('TensorFlow/Keras Model Prediction')

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

 # Make predictions
 if st.button('Predict'):
    if features is not None:
        # Convert features to float32
        features = features.astype(np.float32)
        predictions = model.predict(features)

        # Get the predicted sentiment (index of the maximum probability)
        predicted_classes = np.argmax(predictions, axis=1)

        # Map 0 to 'Negative' and 1 to 'Positive'
        sentiment_map = {0: 'Negative', 1: 'Positive'}
        predicted_sentiments = [sentiment_map[pred] for pred in predicted_classes]

        # Print the predicted sentiments
        st.write(f'Predicted Sentiments: {predicted_sentiments}')

        # Calculate percentages of Negative and Positive predictions
        total_predictions = len(predicted_sentiments)
        negative_count = predicted_sentiments.count('Negative')
        positive_count = predicted_sentiments.count('Positive')

        negative_percentage = (negative_count / total_predictions) * 100
        positive_percentage = (positive_count / total_predictions) * 100

        # Print the percentages
        st.write(f'Negative Sentiment: {negative_percentage:.2f}%')
        st.write(f'Positive Sentiment: {positive_percentage:.2f}%')

        # Print the sentiment with the highest percentage
        if negative_percentage > positive_percentage:
            st.write(f'Overall Predicted Sentiment: Negative ({negative_percentage:.2f}%)')
        else:
            st.write(f'Overall Predicted Sentiment: Positive ({positive_percentage:.2f}%)')

run_app()
