
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "color_cnn_v2.keras"
CLASS_PATH = ROOT / "outputs" / "cnn" / "class_names_v2.json"

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5faf5;
}
h1, h2, h3 {
    color: #174d2a;
}
.hero {
    background: linear-gradient(120deg, #174d2a, #39834d);
    padding: 30px;
    border-radius: 15px;
    color: white;
}
.hero h1, .hero h3, .hero p {
    color: white;
}
.info {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dcebdd;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_cnn():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    if len(classes) != model.output_shape[-1]:
        raise ValueError("Model output and class count do not match.")

    return model, classes


def readable_name(label):
    parts = label.split("___", 1)

    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    return crop, condition


st.markdown("""
<div class="hero">
    <h1>🌿 Plant Disease Detection System</h1>
    <h3>AI-powered plant leaf disease identification</h3>
    <p>Upload a plant leaf image to identify its possible disease.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

menu = st.selectbox(
    "Navigation",
    ["Home", "Disease Detection", "About Project"]
)

if menu == "Home":

    st.title("Welcome to Plant Disease Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info">
        <h3>🌱 Smart Agriculture</h3>
        Identify possible plant leaf diseases
        using machine learning and deep learning.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info">
        <h3>🧠 CNN Technology</h3>
        Our system uses a Convolutional Neural
        Network trained on PlantVillage images.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info("The CNN supports 38 plant disease and healthy-leaf classes.")

    st.subheader("How to use")
    st.write("""
    1. Open Disease Detection.
    2. Upload a JPG, JPEG or PNG leaf image.
    3. Click the Predict Disease button.
    4. View the predicted class and model score.
    """)

elif menu == "Disease Detection":

    st.title("🔍 Plant Disease Detection")
    st.write("Upload a clear image of a plant leaf.")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert("RGB")

            st.image(
                image,
                caption="Uploaded Plant Leaf",
                width=450
            )

            if st.button("🔬 Predict Disease", type="primary"):

                with st.spinner("Loading CNN model and analysing image..."):

                    model, classes = load_cnn()

                    resized = image.resize(
                        (128, 128),
                        Image.Resampling.BILINEAR
                    )

                    image_array = np.asarray(
                        resized,
                        dtype=np.float32
                    ) / 255.0

                    image_array = np.expand_dims(image_array, axis=0)

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    best_index = int(np.argmax(predictions))
                    best_score = float(predictions[best_index])

                    predicted_label = classes[best_index]
                    crop, condition = readable_name(predicted_label)

                st.divider()
                st.subheader("Prediction Result")

                st.success("Image analysis completed!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Plant", crop)

                with col2:
                    st.metric("Predicted Class", condition)

                st.write("**Model score**")
                st.progress(best_score)
                st.write(f"{best_score * 100:.2f}%")

                if "healthy" in condition.lower():
                    st.success(
                        "The model classified this leaf as healthy."
                    )
                else:
                    st.warning(
                        f"The model detected a possible condition: {condition}"
                    )

                st.subheader("Other possible predictions")

                top_indices = np.argsort(predictions)[::-1][:3]

                for index in top_indices:
                    name_crop, name_condition = readable_name(
                        classes[int(index)]
                    )
                    score = float(predictions[int(index)]) * 100

                    st.write(
                        f"**{name_crop} — {name_condition}:** "
                        f"{score:.2f}%"
                    )

                st.info("""
                Important: These scores are model outputs, not confirmed
                diagnoses. Results may be inaccurate, especially for
                unclear images or plants outside the training dataset.
                """)

                if "healthy" not in condition.lower():
                    st.subheader("General precautions")
                    st.write("""
                    - Inspect other leaves for similar symptoms.
                    - Avoid spreading potentially affected plant material.
                    - Keep gardening tools clean.
                    - Consult a local agricultural expert before choosing
                      any disease treatment.
                    """)

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")

    else:
        st.info("Upload a JPG or PNG leaf image to begin.")

elif menu == "About Project":

    st.title("About Our Project")

    st.write("""
    Plant Disease Detection is a machine learning and deep learning
    project designed to classify plant leaf images.

    **Dataset:** PlantVillage

    **Number of classes:** 38

    **Dataset images:** 54,305

    **Models developed:**
    - HOG + Linear SVM
    - HOG + RGB Color Histogram + Linear SVM
    - Convolutional Neural Network (CNN)

    **CNN input size:** 128 × 128 pixels

    **Technologies:**
    - Python
    - TensorFlow
    - NumPy
    - Scikit-learn
    - Streamlit
    - Pillow
    """)

st.divider()

st.caption(
    "Plant Disease Detection | Academic Machine Learning Project"
)
