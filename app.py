import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load trained model
model = tf.keras.models.load_model("plant_model_small.h5")

# Load class names
with open("classes.json") as f:
    classes = json.load(f)

# Remedy suggestions
remedies = {
    "Apple leaf":"Leaf appears healthy. Maintain proper watering and balanced fertilizer.",
    "Apple rust leaf":"Apply copper based fungicide and remove infected leaves.",
    "Bell_pepper leaf":"Plant is healthy. Maintain balanced soil nutrients.",
    "Bell_pepper leaf spot":"Use fungicide and remove infected leaves to prevent spread.",
    "Tomato Early blight leaf":"Remove infected leaves and apply nitrogen fertilizer or fungicide.",
    "Tomato Late blight leaf":"Use potassium fertilizer and apply fungicide to control infection."
}

# App title
st.title("🌿 Plant Nutrient Deficiency Detection System")

st.write("Upload a plant leaf image to detect disease and get remedy suggestion.")

# Upload image
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Leaf Image", width=400)

    # Preprocess image
    img = img.resize((224,224))
    img_array = np.array(img)/255
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    result = classes[index]

    confidence = prediction[0][index] * 100

    # Output
    st.subheader("Detected Condition")
    st.success(result)

    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Recommended Remedy")
    st.info(remedies[result])