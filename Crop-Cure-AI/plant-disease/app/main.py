import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Disease")
    return predicted_class_name

def get_treatment_info(disease_name, db_path):
    with open(db_path, 'r') as f:
        treatment_info = json.load(f)
    return treatment_info.get(disease_name, None)

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            try:
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')

                db_path = f"{working_dir}/Data/treatment_info.json"
                treatment_info = get_treatment_info(prediction, db_path)
                if treatment_info:
                    st.header("Treatment Information")
                    st.write(f"Symptoms: {treatment_info['symptoms']}")
                    st.write(f"Causes: {treatment_info['causes']}")
                    st.write(f"Treatment: {treatment_info['treatment']}")
                else:
                    st.warning("Treatment information not available.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

                st.header("Available Plant Diseases")
                plant_diseases = list(class_indices.values())
                st.write(plant_diseases)
