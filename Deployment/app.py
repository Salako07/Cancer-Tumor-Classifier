import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model
model = tf.keras.models.load_model('Best_model.h5')  # Adjust the path to your model

# Define class labels (adjust these to match your model's output classes)
class_labels = ['Benign', 'Malignant', 'Normal']  # Update with your actual class names

def preprocess_image(image):
    # Preprocess your image here (e.g., resizing, normalization)
    image = image.resize((224, 224))  # Adjust to your model's input size
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    predicted_class = class_labels[predicted_class_index[0]]  # Map index to class label

    # Display the prediction
    st.write("Predicted Class:", predicted_class)
    st.write("Prediction Probabilities:", predictions)
