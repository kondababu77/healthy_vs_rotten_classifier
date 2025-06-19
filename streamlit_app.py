import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load model
@st.cache_resource
def load():
    return load_model("healthy_vs_rotten_model.h5")

model = load()
class_names = ["Healthy", "Rotten"]

st.title("🍓 Healthy vs Rotten Classifier")
st.write("Upload a fruit or vegetable image to check freshness.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = class_names[0] if prediction < 0.5 else class_names[1]
    confidence = 1 - prediction if prediction < 0.5 else prediction

    st.subheader(f"🧠 Prediction: {label}")
    st.write(f"📊 Confidence: {confidence * 100:.2f}%")
