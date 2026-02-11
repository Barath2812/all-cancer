import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Leukemia Detection (ALL vs HEM)",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Leukemia Classification System")
st.markdown("### ALL vs HEM Detection using ResNet50")

# ============================================================
# LOAD MODEL (CACHED)
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "ResNet50_ALL_HEM_FINE_TUNED_FINAL2.keras",
        compile=False
    )
    return model

model = load_model()

# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold (Lower = Higher ALL Recall)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("""
- Lower threshold → Detect more ALL cases  
- Higher threshold → Reduce false positives  
""")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

IMG_SIZE = (224, 224)

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Blood Smear Image (.bmp, .jpg, .png)",
    type=["bmp", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]

        prob_all = float(prediction)
        prob_hem = 1 - prob_all

        predicted_class = "ALL" if prob_all > threshold else "HEM"

        st.subheader("🔍 Prediction Result")

        if predicted_class == "ALL":
            st.error(f"⚠️ Predicted: ALL (Leukemia)")
        else:
            st.success(f"✅ Predicted: HEM (Healthy)")

        st.write(f"Confidence (ALL): {prob_all:.4f}")
        st.write(f"Confidence (HEM): {prob_hem:.4f}")

        # Probability Bar
        st.subheader("📊 Prediction Probabilities")

        st.progress(prob_all)
        st.write(f"ALL Probability: {prob_all:.2%}")

        st.progress(prob_hem)
        st.write(f"HEM Probability: {prob_hem:.2%}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
⚠️ **Disclaimer:**  
This AI system is for research/educational purposes only.  
Not intended for clinical diagnosis.
""")
