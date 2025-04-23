import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Swinv2ForImageClassification

# Page setup
st.set_page_config(page_title="Glaucoma Detector", layout="centered")
st.title("👁️ Glaucoma Detection App")
st.write("Upload a **fundus image** to check for signs of glaucoma using a deep learning model.")

# File uploader
uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Analyzing the image..."):
        image_np = np.array(image)

        processor = AutoImageProcessor.from_pretrained("pamixsun/swinv2_tiny_for_glaucoma_classification")
        model = Swinv2ForImageClassification.from_pretrained("pamixsun/swinv2_tiny_for_glaucoma_classification")

        inputs = processor(image_np, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]

        st.success(f"✅ **Prediction:** {label}")
