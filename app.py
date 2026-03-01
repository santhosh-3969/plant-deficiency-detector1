import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Plant Deficiency Detector", layout="centered")

# Custom CSS
st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#2e7d32;
}
.sub-title{
    text-align:center;
    color:gray;
}
.result-box{
    padding:20px;
    border-radius:10px;
    background-color:#f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("plant_model_small.h5")

# Load classes
with open("classes.json") as f:
    classes = json.load(f)

# Remedies
remedies = {
"Apple leaf":"Plant is healthy. Maintain proper watering and nutrients.",
"Apple rust leaf":"Apply copper fungicide and remove infected leaves.",
"Bell_pepper leaf":"Healthy plant. Maintain balanced fertilizer.",
"Bell_pepper leaf spot":"Use fungicide and remove infected leaves.",
"Tomato Early blight leaf":"Apply fungicide and remove affected leaves.",
"Tomato Late blight leaf":"Apply fungicide and avoid excess moisture."
}

# Title
st.markdown("<div class='main-title'>🌿 Plant Nutrient Deficiency Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload a leaf image to detect disease and get remedy</div>", unsafe_allow_html=True)

st.write("")

# Upload
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Leaf", width=350)

    # preprocess
    img = img.resize((224,224))
    img_array = np.array(img)/255
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    result = classes[index]

    confidence = prediction[0][index]*100

    st.write("")

    # Result Card
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    st.subheader("🔍 Prediction Result")
    st.success(result)

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("💊 Recommended Remedy")
    st.info(remedies[result])

    st.markdown("</div>", unsafe_allow_html=True)
