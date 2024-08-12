import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up working directory and paths
working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_data
def load_class_indices():
    class_indices_path = os.path.join(working_dir, "class_indices.json")
    with open(class_indices_path, "r") as f:
        return json.load(f)

@st.cache_data
def preprocess_image(image, target_size=(224, 224)):  # Ensure target size matches model input
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image, class_indices):
    try:
        preprocessed_img = preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Disease")
        return predicted_class_name
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def get_treatment_info(disease_name, db_path):
    with open(db_path, 'r') as f:
        treatment_info = json.load(f)
    return treatment_info.get(disease_name, None)

# Streamlit App
st.set_page_config(page_title="Crop-Care-AI", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);  /* Light gradient background */
        color: #2c3e50;  /* Dark text color */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;  /* Modern font */
    }
    .stSidebar {
        background-color: #ffffff;  /* White sidebar background */
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);  /* Subtle shadow for sidebar */
    }
    .stButton > button {
        background-color: #1abc9c;  /* Teal button background */
        color: #ffffff;  /* White text color for buttons */
        border-radius: 12px;  /* Rounded button edges */
        padding: 10px 20px;  /* Padding for buttons */
        transition: background-color 0.3s ease, transform 0.3s ease;  /* Smooth transition for hover effect */
    }
    .stButton > button:hover {
        background-color: #16a085;  /* Darker teal on hover */
        transform: scale(1.05);  /* Slightly larger button on hover */
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;  /* White background for text input */
        color: #2c3e50;  /* Dark text color for input */
        border-radius: 8px;  /* Rounded input edges */
        padding: 10px;  /* Padding for text input */
        border: 1px solid #dcdde1;  /* Border for input */
    }
    .stMarkdown {
        line-height: 1.6;  /* Improved readability */
    }
    .main-container {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-top: 20px;
    }
    .image-container {
        flex: 1;
        margin-right: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px solid #ddd;  /* Border for visual clarity */
        border-radius: 10px;
        padding: 10px;
        height: 300px;  /* Fixed height for equal sizing */
    }
    .image-container img {
        height: 100%;  /* Fill the container height */
        width: 100%;  /* Fill the container width */
        object-fit: cover;  /* Ensure image covers the container */
    }
    .info-container {
        flex: 1;
        border-left: 4px solid #1abc9c;
        padding: 15px;
        background: #ecf0f1;
        border-radius: 5px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
    }
    .info-container h4 {
        margin-top: 0;  /* Remove top margin from heading */
    }
    .info-content {
        display: flex;
        flex-direction: column;
        margin-top: 10px;  /* Space between heading and content */
    }
    .info-content > p {
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🌱 Crop-Care-AI')
st.markdown("""
    <p style='font-size: 18px; line-height: 1.6;'>
        Welcome to Crop-Care-AI, your ultimate tool for diagnosing plant diseases.<br>
        Upload an image of your plant to get accurate predictions and treatment suggestions.
    </p>
    """, unsafe_allow_html=True)

# Load the model and class indices
model = load_model()
class_indices = load_class_indices()

# File uploader
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([1, 1])  # Equal width columns

    with col1:
        st.subheader("Uploaded Image")
        # Resize image to be square and match treatment info box height
        image = image.convert("RGB")  # Ensure the image is in RGB mode
        image = image.resize((300, 300))  # Resize to a fixed square size
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button('Classify'):
            try:
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.markdown(f"<h3>Prediction: {str(prediction)}</h3>", unsafe_allow_html=True)

                db_path = os.path.join(working_dir, "Data", "treatment_info.json")
                treatment_info = get_treatment_info(prediction, db_path)
                if treatment_info:
                    st.markdown("""
                    <div class='info-container'>
                        <h4>Treatment Information</h4>
                        <div class='info-content'>
                            <p><b>Symptoms:</b> {symptoms}</p>
                            <p><b>Causes:</b> {causes}</p>
                            <p><b>Treatment:</b> {treatment}</p>
                        </div>
                    </div>
                    """.format(**treatment_info), unsafe_allow_html=True)
                else:
                    st.warning("Treatment information not available.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.header("Available Plant Diseases")
                plant_diseases = list(class_indices.values())
                st.write(plant_diseases)
