
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# Load the trained TensorFlow/Keras model
model_path = '/content/drive/MyDrive/Colab Notebooks/SaveModel/model.h5'
model = tf.keras.models.load_model(model_path)
# Streamlit app code
def run_app():
    st.title('TensorFlow/Keras Model Prediction')

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        # Load the CSV using pandas
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.write("Data Loaded Successfully:")
        st.write(data)
        # Convert data to numpy array (without the labels)
        features = data.iloc[:, 1:].values  # Assuming the first column is a label or ID


        # Make predictions
        if st.button('Predict'):
            predictions = model.predict(data)
            st.write(f'Predictions: {predictions}')


run_app()
