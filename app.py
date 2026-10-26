import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model

class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
model = load_model("emotion_model.keras")

enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    img = Image.open(picture).convert("L").resize((48, 48))
    x = np.array(img, dtype="float32") / 255.0
    x = x.reshape(1, 48, 48, 1)

    probs = model.predict(x)[0]
    st.write(class_names[probs.argmax()], f"{probs.max():.0%}")