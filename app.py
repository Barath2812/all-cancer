# ============================================================
# 🩸 Leukemia Classification (ALL vs HEM) – Streamlit App
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "Leukemia_ALL_HEM_FINAL_BALANCED.keras"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Leukemia Detection",
    layout="centered"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    if img.shape[-1] == 4:  # remove alpha if present
        img = img[..., :3]
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# UI
# -------------------------------
st.title("🩸 Leukemia Detection System")
st.markdown("**Binary Classification: ALL vs HEM**")

st.markdown("""
This system uses a **deep learning EfficientNet model** to classify blood smear images.

**Modes:**
- 🟢 *Screening Mode*: High sensitivity for ALL
- 🔵 *Confirmation Mode*: High precision
""")

mode = st.radio(
    "Select Operating Mode",
    ["Screening (High ALL Recall)", "Confirmation (High Precision)"]
)

threshold = 0.35 if "Screening" in mode else 0.50

uploaded_file = st.file_uploader(
    "Upload Blood Smear Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed = preprocess_image(image)
            prob = model.predict(processed)[0][0]

            prediction = "ALL (Leukemia)" if prob > threshold else "HEM (Healthy)"
            confidence = prob if prob > threshold else 1 - prob

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prediction.startswith("ALL"):
            st.error(f"🧬 **Prediction:** {prediction}")
        else:
            st.success(f"✅ **Prediction:** {prediction}")

        st.metric(
            label="Confidence Score",
            value=f"{confidence * 100:.2f}%"
        )

        st.caption(f"Decision Threshold Used: {threshold}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption(
    "⚠️ This system is for educational and research purposes only. "
    "Not intended for clinical diagnosis."
)
