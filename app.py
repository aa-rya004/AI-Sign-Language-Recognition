
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="AI Sign Language Recognition",
    page_icon="🤟"
)

st.title("🤟 AI Sign Language Recognition")
st.write("Recognize American Sign Language alphabets A-Z using Deep Learning.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_model.h5")

model = load_model()

labels = [chr(i) for i in range(65, 91)]
IMG_SIZE = 64

picture = st.camera_input("Show your hand gesture")

if picture is not None:
    image = Image.open(picture).convert("RGB")
    img = np.array(image)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence > 0.7:
        st.success(f"Predicted Alphabet: {labels[class_index]}")
    else:
        st.warning("Unable to recognize the sign clearly.")

    st.write(f"Confidence: {confidence * 100:.2f}%")
