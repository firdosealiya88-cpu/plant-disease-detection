import io
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import requests
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_plantcare.keras"
CLASS_PATH = ROOT / "outputs" / "mobilenet" / "class_names_mobilenet.json"
DATA_DIR = ROOT / "data"
HISTORY_PATH = DATA_DIR / "scan_history.csv"

DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
.stApp {background: #f5f8f4;}
section[data-testid="stSidebar"] {background: #eaf2e8;}
.hero {
    padding: 28px; border-radius: 18px;
    background: linear-gradient(120deg,#14532d,#4d9b61);
    color: white; margin-bottom: 20px;
}
.hero h1 {color:white; font-size:38px;}
.hero p {color:#e8f5e9; font-size:17px;}
.metric-card {
    background:white; padding:18px; border-radius:14px;
    border:1px solid #dce8dc;
}
</style>
""", unsafe_allow_html=True)

# ---------- Language ----------
LANG = {
    "English": {
        "home": "Home", "detect": "Disease Detection",
        "history": "Detection History", "analytics": "Analytics Dashboard",
        "library": "Disease Library", "doctor": "Plant Doctor",
        "risk": "Environmental Risk", "map": "Crop Disease Risk Map",
        "about": "About Project", "upload": "Upload plant leaf images",
        "predict": "Analyze image", "healthy": "Healthy",
        "diseased": "Possible disease detected"
    },
    "Hindi": {
        "home": "होम", "detect": "रोग पहचान",
        "history": "पहचान इतिहास", "analytics": "विश्लेषण डैशबोर्ड",
        "library": "रोग पुस्तकालय", "doctor": "प्लांट डॉक्टर",
        "risk": "पर्यावरणीय जोखिम", "map": "फसल रोग जोखिम मानचित्र",
        "about": "परियोजना के बारे में", "upload": "पत्तियों की तस्वीरें अपलोड करें",
        "predict": "तस्वीर का विश्लेषण करें", "healthy": "स्वस्थ",
        "diseased": "संभावित रोग का पता चला"
    },
    "Kannada": {
        "home": "ಮುಖಪುಟ", "detect": "ರೋಗ ಪತ್ತೆ",
        "history": "ಪತ್ತೆ ಇತಿಹಾಸ", "analytics": "ವಿಶ್ಲೇಷಣೆ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "library": "ರೋಗ ಗ್ರಂಥಾಲಯ", "doctor": "ಪ್ಲಾಂಟ್ ಡಾಕ್ಟರ್",
        "risk": "ಪರಿಸರ ಅಪಾಯ", "map": "ಬೆಳೆ ರೋಗ ಅಪಾಯ ನಕ್ಷೆ",
        "about": "ಯೋಜನೆಯ ಬಗ್ಗೆ", "upload": "ಎಲೆಗಳ ಚಿತ್ರಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "predict": "ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ", "healthy": "ಆರೋಗ್ಯಕರ",
        "diseased": "ಸಂಭಾವ್ಯ ರೋಗ ಪತ್ತೆಯಾಗಿದೆ"
    }
}

# ---------- Model ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Class file not found: {CLASS_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    if isinstance(classes, dict):
        classes = list(classes.values())

    return model, list(classes)

def read_history():
    if HISTORY_PATH.exists() and HISTORY_PATH.stat().st_size > 0:
        try:
            return pd.read_csv(HISTORY_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_history(record):
    df = read_history()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)

def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize((128, 128))
    # MobileNetV2 preprocessing is included inside the trained model.
    array = np.asarray(resized, dtype=np.float32)
    return image, np.expand_dims(array, axis=0)

def quality_check(image):
    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    sharpness = float(np.var(
        gray[1:, :] - gray[:-1, :]
    ) + np.var(gray[:, 1:] - gray[:, :-1]))

    notes = []
    if min(width, height) < 128:
        notes.append("Image resolution is low.")
    if brightness < 45:
        notes.append("Image may be too dark.")
    elif brightness > 220:
        notes.append("Image may be overexposed.")
    if sharpness < 100:
        notes.append("Image may be blurry.")

    return {
        "width": width, "height": height,
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "notes": notes
    }

def decode_class(name):
    parts = str(name).split("___", 1)
    crop = parts[0].replace("_", " ").replace(",", "")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    disease = disease.strip()
    return crop, disease

def precautions(disease):
    d = disease.lower()
    if "healthy" in d:
        return [
            "Continue regular watering appropriate for the crop.",
            "Inspect leaves regularly for changes.",
            "Keep the growing area clean and well ventilated."
        ]
    if any(x in d for x in ["virus", "mosaic", "curl"]):
        return [
            "Separate plants showing suspicious symptoms where practical.",
            "Clean gardening tools between plants.",
            "Monitor for insects that may spread plant diseases.",
            "Seek local agricultural guidance for suspected viral disease."
        ]
    if any(x in d for x in ["bacterial", "spot"]):
        return [
            "Avoid wetting foliage when watering.",
            "Remove badly affected fallen plant material safely.",
            "Allow good spacing and airflow between plants.",
            "Ask a local agricultural expert before choosing treatment."
        ]
    if any(x in d for x in ["mildew", "blight", "scab", "rot", "rust"]):
        return [
            "Remove fallen and badly affected plant material.",
            "Improve airflow around plants.",
            "Avoid excessive moisture on leaves.",
            "Use only locally approved crop treatments as directed."
        ]
    return [
        "Inspect the plant and nearby plants for similar symptoms.",
        "Keep leaves as dry as practical and improve airflow.",
        "Remove fallen plant debris.",
        "Consult a local agricultural expert for confirmation."
    ]

def create_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 55

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(45, y, "PlantCare AI - Plant Health Report")
    y -= 35

    pdf.setFont("Helvetica", 11)
    lines = [
        f"Date: {result.get('date', '')}",
        f"Image: {result.get('image', '')}",
        f"Crop predicted: {result.get('crop', '')}",
        f"Condition predicted: {result.get('disease', '')}",
        f"Model score: {result.get('score', 0):.2f}%",
        f"Severity estimate entered by user: {result.get('severity', 'Not entered')}",
        f"User-estimated health score: {result.get('health_score', 'Not calculated')}",
        "",
        "Suggested precautions:"
    ]

    for line in lines:
        pdf.drawString(45, y, str(line)[:100])
        y -= 20

    for item in result.get("precautions", []):
        pdf.drawString(60, y, "- " + item[:95])
        y -= 19

    y -= 15
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        45, y,
        "This AI result is an estimate, not a confirmed agricultural diagnosis."
    )
    pdf.save()
    buffer.seek(0)
    return buffer

def risk_assessment(humidity, rain, temperature, leaf_wetness):
    score = 0
    reasons = []

    if humidity >= 80:
        score += 30
        reasons.append("High humidity")
    elif humidity >= 65:
        score += 15
        reasons.append("Moderately high humidity")

    if rain == "Yes":
        score += 25
        reasons.append("Recent or ongoing rain")

    if leaf_wetness == "Yes":
        score += 25
        reasons.append("Leaves are frequently wet")

    if 20 <= temperature <= 30:
        score += 20
        reasons.append("Temperature may favor some common diseases")

    score = min(score, 100)
    if score >= 60:
        level = "Higher estimated risk"
    elif score >= 30:
        level = "Moderate estimated risk"
    else:
        level = "Lower estimated risk"

    return score, level, reasons

# ---------- Sidebar ----------
st.sidebar.title("🌿 PlantCare AI")
language = st.sidebar.selectbox("Language / भाषा / ಭಾಷೆ", list(LANG.keys()))
T = LANG[language]

mode = st.sidebar.radio("User mode", ["Farmer Mode", "Expert Mode"])
page_options = [
    T["home"],
    T["detect"],
    T["history"],
    T["analytics"],
    T["library"],
    T["doctor"],
    T["risk"],
    T["map"],
    "Plant Progress Tracker",
    "7-Day Crop Care Planner",
    "Agriculture Chat Assistant",
    "Crop Intelligence Dashboard",
    T["about"]
]
page = st.sidebar.radio("Navigation", page_options)

# ---------- Header ----------
st.markdown("""
<div class="hero">
<h1>🌿 PlantCare AI</h1>
<p>AI-assisted plant disease detection and plant health guidance</p>
</div>
""", unsafe_allow_html=True)


# ---------- Home ----------
if page == T["home"]:
    history = read_history()

    st.markdown("## 🌱 PlantCare AI Command Center")
    st.write("Your plant health, disease monitoring and environmental overview.")

    # Dashboard calculations
    total = len(history)
    healthy = 0
    diseased = 0
    crop_count = 0
    avg_score = 0.0

    if not history.empty:
        if "disease" in history.columns:
            healthy = int(
                history["disease"].astype(str)
                .str.lower().str.contains("healthy").sum()
            )
            diseased = total - healthy

        if "crop" in history.columns:
            crop_count = int(history["crop"].nunique())

        if "score" in history.columns:
            scores = pd.to_numeric(history["score"], errors="coerce")
            if scores.notna().any():
                avg_score = float(scores.mean())

    st.markdown("### 📊 Plant Health Overview")
    a, b, c, d = st.columns(4)
    a.metric("Total scans", total)
    b.metric("Healthy predictions", healthy)
    c.metric("Possible disease predictions", diseased)
    d.metric("Crops monitored", crop_count)

    e, f = st.columns(2)
    e.metric("Average model output score", f"{avg_score:.1f}%")
    f.metric("MobileNetV2 supported classes", 38)

    st.caption(
        "Dashboard statistics are calculated from your saved scans. "
        "CNN scores are model outputs, not guaranteed probabilities."
    )

    st.divider()
    st.markdown("### 🌦️ Environmental Monitoring")
    st.write(
        "Enter current conditions from your farm or garden to estimate "
        "whether the environment may favor some plant diseases."
    )

    with st.container(border=True):
        env1, env2 = st.columns(2)
        with env1:
            env_city = st.text_input(
                "Farm or garden location (optional)",
                placeholder="Enter city or area"
            )
            env_temp = st.slider(
                "Temperature (°C)", 0, 50, 25, key="home_temp"
            )
            env_humidity = st.slider(
                "Humidity (%)", 0, 100, 60, key="home_humidity"
            )
        with env2:
            env_rain = st.selectbox(
                "Recent or ongoing rainfall",
                ["No", "Yes"],
                key="home_rain"
            )
            env_wet = st.selectbox(
                "Are leaves frequently wet?",
                ["No", "Yes"],
                key="home_wet"
            )
            st.write("These are manually entered conditions, not live weather readings.")

        if st.button("Calculate environmental risk", type="primary"):
            risk_score, risk_level, risk_reasons = risk_assessment(
                env_humidity, env_rain, env_temp, env_wet
            )
            st.session_state["home_risk_score"] = risk_score
            st.session_state["home_risk_level"] = risk_level
            st.session_state["home_risk_reasons"] = risk_reasons
            st.session_state["home_risk_city"] = env_city

        if "home_risk_score" in st.session_state:
            risk_score = st.session_state["home_risk_score"]
            risk_level = st.session_state["home_risk_level"]
            st.progress(risk_score / 100)
            r1, r2 = st.columns(2)
            r1.metric("Estimated environmental risk", f"{risk_score}/100")
            r2.metric("Risk category", risk_level)

            if risk_score >= 60:
                st.warning("Conditions may favor some plant diseases. Inspect plants regularly.")
            elif risk_score >= 30:
                st.info("Keep monitoring plants and avoid unnecessary leaf wetness.")
            else:
                st.success("Fewer of the selected environmental risk factors are present.")

            reasons = st.session_state["home_risk_reasons"]
            if reasons:
                st.write("Factors:")
                for reason in reasons:
                    st.write("• " + reason)

            st.caption(
                "Rule-based educational estimate only. It is not a live weather "
                "forecast or a validated disease prediction."
            )

    st.divider()
    st.markdown("### 📈 Disease Intelligence")

    if history.empty:
        st.info("Your charts will appear here after you analyze your first plant image.")
    else:
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("**Most frequently predicted conditions**")
            if "disease" in history.columns:
                disease_counts = history["disease"].value_counts().head(8)
                st.bar_chart(disease_counts)

        with chart2:
            st.markdown("**Scans by crop**")
            if "crop" in history.columns:
                crop_counts = history["crop"].value_counts()
                st.bar_chart(crop_counts)

        if "date" in history.columns:
            st.markdown("**Detection activity over time**")
            scan_dates = pd.to_datetime(history["date"], errors="coerce")
            daily_counts = scan_dates.dt.date.value_counts().sort_index()
            if not daily_counts.empty:
                st.line_chart(daily_counts)

    st.divider()
    st.markdown("### 🕒 Recent Plant Scans")

    if history.empty:
        st.info("No recent scans yet. Open Disease Detection to get started.")
    else:
        st.dataframe(
            history.tail(5).iloc[::-1],
            width="stretch",
            hide_index=True
        )

    st.divider()
    st.markdown("### ⚡ Quick Access")

    q1, q2, q3, q4 = st.columns(4)
    q1.markdown("**📷 Disease Detection**")
    q1.caption("Analyze leaf images")
    q2.markdown("**🩺 Plant Doctor**")
    q2.caption("Get plant-care suggestions")
    q3.markdown("**📚 Disease Library**")
    q3.caption("Explore supported diseases")
    q4.markdown("**📄 Reports**")
    q4.caption("Review history and download CSV")

    st.info("Use the navigation menu in the left sidebar to open each feature.")


# LIVE WEATHER SECTION
    @st.cache_data(ttl=900)
    def get_live_weather(city):
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15
        )
        geo_response.raise_for_status()
        places = geo_response.json().get("results", [])

        if not places:
            return None

        place = places[0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,rain,weather_code,cloud_cover,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": 3,
                "timezone": "auto"
            },
            timeout=15
        )
        weather_response.raise_for_status()

        return {
            "place": place,
            "weather": weather_response.json()
        }


    def weather_description(code):
        descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Freezing drizzle",
            57: "Heavy freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail"
        }
        return descriptions.get(code, "Weather information unavailable")

# ---------- Disease Detection ----------
elif page == T["detect"]:
    st.subheader("🌿 AI Plant Disease Detection")
    st.write("Upload a plant leaf image or capture one using your camera.")

    plant_id = st.text_input(
        "Plant ID (use the same ID for future scans)",
        placeholder="Example: TOMATO-001"
    )

    uploaded_file = st.file_uploader(
        T["upload"],
        type=["jpg", "jpeg", "png"],
        key="disease_upload"
    )

    camera_file = st.camera_input(
        "Or capture a leaf image",
        key="disease_camera"
    )

    image_file = uploaded_file if uploaded_file is not None else camera_file

    if image_file is not None:
        try:
            original_image = Image.open(image_file)
            original_image, input_array = prepare_image(original_image)

            st.image(
                original_image,
                caption="Selected plant leaf image",
                width=400
            )

            quality = quality_check(original_image)

            st.markdown("### Image Quality")
            q1, q2, q3 = st.columns(3)
            q1.metric("Width", quality["width"])
            q2.metric("Height", quality["height"])
            q3.metric("Sharpness", quality["sharpness"])

            for note in quality["notes"]:
                st.warning(note)

            severity = st.selectbox(
                "Your estimated symptom severity",
                ["Not entered", "Mild", "Moderate", "Severe"]
            )

            if st.button("🔍 Analyze Plant Image", type="primary"):
                with st.spinner("Loading CNN model and analyzing image..."):
                    model, class_names = load_model()

                    predictions = np.asarray(
                        model.predict(input_array, verbose=0)
                    ).reshape(-1)

                    if len(predictions) != len(class_names):
                        st.error(
                            "Model output count does not match the class names. "
                            "Please check your model and class_names_v2.json."
                        )
                    else:
                        # Convert logits to probabilities if necessary.
                        if (
                            np.any(predictions < 0)
                            or not np.isclose(predictions.sum(), 1.0, atol=0.05)
                        ):
                            exp_values = np.exp(
                                predictions - np.max(predictions)
                            )
                            predictions = exp_values / exp_values.sum()

                        best_index = int(np.argmax(predictions))
                        class_name = class_names[best_index]
                        crop, disease = decode_class(class_name)
                        score = float(predictions[best_index] * 100)

                        top_indices = np.argsort(predictions)[-3:][::-1]
                        top_predictions = [
                            {
                                "class": class_names[int(i)],
                                "score": float(predictions[int(i)] * 100)
                            }
                            for i in top_indices
                        ]

                        is_healthy = "healthy" in disease.lower()

                        result = {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "plant_id": plant_id.strip() or "Not provided",
                            "image": image_file.name,
                            "crop": crop,
                            "disease": disease,
                            "score": round(score, 2),
                            "top_predictions": top_predictions,
                            "severity": severity,
                            "precautions": precautions(disease)
                        }

                        save_history(result)
                        st.session_state["latest_prediction"] = result

            if "latest_prediction" in st.session_state:
                result = st.session_state["latest_prediction"]

                st.divider()
                st.markdown("### 🧪 Prediction Result")

                if "healthy" in result["disease"].lower():
                    st.success(f"🌱 {result['crop']}: {result['disease']}")
                else:
                    st.warning(
                        f"Possible condition: {result['disease']}"
                    )

                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted crop", result["crop"])
                r2.metric("Predicted condition", result["disease"])
                r3.metric("Model output score", f"{result['score']:.2f}%")

                st.markdown("### Top 3 model outputs")
                for item in result.get("top_predictions", []):
                    top_crop, top_disease = decode_class(item["class"])
                    st.write(
                        f"**{top_crop} ? {top_disease}** "
                        f"({item['score']:.2f}%)"
                    )
                    st.progress(
                        min(1.0, max(0.0, item["score"] / 100))
                    )

                st.warning(
                    "EXPERIMENTAL MODEL: These are unverified AI estimates. "
                    "MobileNetV2 achieved 90.41% accuracy on 8,146 held-out test images. "
                    "Do not use these results alone to decide treatment. "
                    "Confirm with an agricultural expert."
                )

                st.caption(
                    "The model output score is not a guaranteed probability "
                    "or a confirmed agricultural diagnosis."
                )

                st.markdown("### 🌿 Suggested Precautions")
                for tip in result["precautions"]:
                    st.write("• " + tip)

                pdf_buffer = create_pdf(result)
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name="PlantCare_AI_Report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Image analysis failed: {e}")



# ---------- Detection History ----------
elif page == T["history"]:
    st.subheader("📋 Detection History")
    history = read_history()

    if history.empty:
        st.info("No plant scans have been saved yet.")
    else:
        st.metric("Total saved scans", len(history))

        if "date" in history.columns:
            history["date"] = pd.to_datetime(
                history["date"], errors="coerce"
            )

        st.dataframe(
            history.sort_values("date", ascending=False)
            if "date" in history.columns else history,
            width="stretch",
            hide_index=True
        )

        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scan history CSV",
            csv_data,
            "plantcare_scan_history.csv",
            "text/csv"
        )


# ---------- Plant Progress Tracker ----------
elif page == "Plant Progress Tracker":
    st.subheader("🌱 Plant Progress Tracker")
    st.write("Compare previous scans of the same plant using its Plant ID.")

    history = read_history()

    if history.empty or "plant_id" not in history.columns:
        st.info("No Plant ID records found. Analyze a leaf image and enter a Plant ID first.")
    else:
        available_ids = sorted([
            str(x) for x in history["plant_id"].dropna().unique()
            if str(x).strip() and str(x) != "Not provided"
        ])

        if not available_ids:
            st.info("No named Plant IDs found. Enter a Plant ID during your next scan.")
        else:
            selected_id = st.selectbox("Select Plant ID", available_ids)

            plant_data = history[
                history["plant_id"].astype(str) == selected_id
            ].copy()

            if "date" in plant_data.columns:
                plant_data["date"] = pd.to_datetime(
                    plant_data["date"], errors="coerce"
                )
                plant_data = plant_data.sort_values("date")

            st.markdown(f"### Progress report: {selected_id}")

            p1, p2, p3 = st.columns(3)
            p1.metric("Total scans", len(plant_data))
            p2.metric(
                "Latest condition",
                str(plant_data.iloc[-1].get("disease", "Unknown"))
            )
            p3.metric(
                "Latest model score",
                f"{pd.to_numeric(pd.Series([plant_data.iloc[-1].get('score', 0)]), errors='coerce').iloc[0]:.1f}%"
            )

            if len(plant_data) >= 2 and "score" in plant_data.columns:
                plant_data["score"] = pd.to_numeric(
                    plant_data["score"], errors="coerce"
                )
                st.markdown("### Model output score across scans")
                chart_data = plant_data.dropna(subset=["score"])

                if not chart_data.empty:
                    if "date" in chart_data.columns:
                        st.line_chart(
                            chart_data.set_index("date")["score"]
                        )
                    else:
                        st.line_chart(chart_data["score"])

                st.caption(
                    "Changes in model scores do not necessarily indicate "
                    "that the plant has improved or worsened."
                )
            else:
                st.info("Complete at least two scans with this Plant ID to compare scan history.")

            st.markdown("### Previous scans")
            st.dataframe(
                plant_data,
                width="stretch",
                hide_index=True
            )


# ---------- 7-Day Crop Care Planner ----------
elif page == "7-Day Crop Care Planner":
    st.subheader("🗓️ 7-Day Crop Care Planner")
    st.write("Create a general plant-care schedule based on your crop and condition.")

    history = read_history()

    crop_options = ["Tomato", "Potato", "Pepper", "Apple", "Grape", "Corn", "Other"]

    if not history.empty and "crop" in history.columns:
        saved_crops = sorted(
            history["crop"].dropna().astype(str).unique().tolist()
        )
        crop_options = list(dict.fromkeys(saved_crops + crop_options))

    selected_crop = st.selectbox("Select crop", crop_options)
    condition = st.selectbox(
        "Current plant condition",
        [
            "Healthy",
            "Possible fungal disease",
            "Possible bacterial disease",
            "Possible viral disease",
            "Unknown symptoms"
        ]
    )

    care_tasks = [
        "Inspect leaves and note any visible changes.",
        "Check soil moisture before watering.",
        "Remove fallen plant debris around the plant.",
        "Inspect both sides of leaves for unusual symptoms.",
        "Check drainage and airflow around the plant.",
        "Review the plant for changes since the first day.",
        "Record a new leaf scan if symptoms have changed."
    ]

    if "fungal" in condition.lower():
        care_tasks[2] = "Check for leaf spots or powdery growth and remove fallen debris."
        care_tasks[4] = "Check airflow and avoid unnecessary leaf wetness."
    elif "bacterial" in condition.lower():
        care_tasks[2] = "Remove fallen affected plant material safely."
        care_tasks[4] = "Avoid wetting foliage and review plant spacing."
    elif "viral" in condition.lower():
        care_tasks[2] = "Keep tools clean and avoid moving suspicious plant material between plants."
        care_tasks[4] = "Inspect nearby plants for similar symptoms."
    elif condition == "Healthy":
        care_tasks[2] = "Keep the growing area clean."
        care_tasks[4] = "Check that the plant has suitable growing conditions."

    if st.button("Generate 7-Day Care Plan", type="primary"):
        st.session_state["care_plan"] = {
            "crop": selected_crop,
            "condition": condition,
            "tasks": care_tasks
        }

    if "care_plan" in st.session_state:
        plan = st.session_state["care_plan"]

        st.markdown(
            f"### Care schedule for {plan['crop']}"
        )
        st.caption(f"Selected condition: {plan['condition']}")

        for day, task in enumerate(plan["tasks"], start=1):
            with st.container(border=True):
                st.markdown(f"**Day {day}**")
                st.write(task)

        st.info(
            "This is a general educational schedule, not a crop-specific "
            "treatment prescription. Adapt it to local agricultural advice."
        )


# ---------- Agriculture Chat Assistant ----------
elif page == "Agriculture Chat Assistant":
    st.subheader("💬 Agriculture Chat Assistant")
    st.write(
        "Ask common plant-care questions. This assistant uses predefined "
        "educational answers and is not a generative AI diagnosis system."
    )

    question = st.text_input(
        "Enter your plant-care question",
        placeholder="Example: How can I prevent fungal diseases?"
    )

    if st.button("Get Answer", type="primary"):
        q = question.lower().strip()

        if not q:
            st.warning("Please enter a question.")
        else:
            answers = [
                (["yellow", "yellowing"],
                 "Yellow leaves can be associated with watering, drainage, "
                 "nutrient or disease issues. Check soil moisture and inspect "
                 "whether older or newer leaves are affected."),
                (["fungal", "fungus", "mildew"],
                 "Improve airflow, avoid unnecessary leaf wetness and remove "
                 "fallen plant debris. Seek local agricultural advice if symptoms spread."),
                (["water", "watering", "irrigation"],
                 "Check soil moisture before watering. Water needs vary by crop, "
                 "growth stage, soil and weather."),
                (["healthy", "health"],
                 "Inspect plants regularly, maintain suitable watering, "
                 "keep the growing area clean and monitor new growth."),
                (["pest", "insect"],
                 "Inspect both sides of leaves and monitor for visible pests. "
                 "Use locally approved management advice."),
                (["tomato"],
                 "For tomatoes, monitor leaves and stems regularly, provide "
                 "suitable spacing and avoid unnecessary foliage wetness."),
                (["potato"],
                 "For potatoes, inspect leaves regularly for spots, unusual "
                 "discoloration and changes in plant growth."),
                (["soil", "drainage"],
                 "Good drainage helps prevent waterlogging. Check that water "
                 "can drain from the growing area."),
                (["scan", "image", "photo", "detect"],
                 "Open Disease Detection, upload a clear leaf image and select "
                 "Analyze Plant Image to see the model's prediction.")
            ]

            response = (
                "I could not find a matching answer. Please describe the crop "
                "and symptoms, or consult a local agricultural expert."
            )

            for keywords, answer in answers:
                if any(word in q for word in keywords):
                    response = answer
                    break

            st.markdown("### Assistant response")
            st.info(response)

    st.caption(
        "This is a rule-based FAQ helper. Its answers are general guidance, "
        "not professional agricultural diagnosis."
    )


# ---------- Crop Intelligence Dashboard ----------
elif page == "Crop Intelligence Dashboard":
    st.subheader("📊 Crop Intelligence Dashboard")
    st.write("Explore crop-wise disease detection statistics from saved scans.")

    history = read_history()

    if history.empty or "crop" not in history.columns:
        st.info("Analyze plant images to generate crop intelligence statistics.")
    else:
        crop_data = history.copy()
        crop_data["crop"] = crop_data["crop"].fillna("Unknown").astype(str)

        total_scans = len(crop_data)
        unique_crops = crop_data["crop"].nunique()

        if "disease" in crop_data.columns:
            healthy_count = int(
                crop_data["disease"].astype(str)
                .str.lower().str.contains("healthy").sum()
            )
        else:
            healthy_count = 0

        diseased_count = total_scans - healthy_count

        c1, c2, c3 = st.columns(3)
        c1.metric("Total crop scans", total_scans)
        c2.metric("Different crops", unique_crops)
        c3.metric("Possible disease predictions", diseased_count)

        st.markdown("### Scans by crop")
        crop_counts = crop_data["crop"].value_counts()
        st.bar_chart(crop_counts)

        if "disease" in crop_data.columns:
            st.markdown("### Predicted conditions by crop")

            selected_crop = st.selectbox(
                "Choose a crop",
                sorted(crop_data["crop"].unique())
            )

            selected_data = crop_data[
                crop_data["crop"] == selected_crop
            ]

            disease_counts = selected_data["disease"].value_counts()
            st.bar_chart(disease_counts)

            st.markdown("### Crop statistics")
            crop_summary = (
                crop_data.groupby("crop")
                .agg(
                    Total_Scans=("disease", "count"),
                    Different_Conditions=("disease", "nunique")
                )
                .reset_index()
            )
            st.dataframe(
                crop_summary,
                width="stretch",
                hide_index=True
            )

        st.caption(
            "Statistics describe saved model predictions only. They do not "
            "represent actual disease prevalence in farms or regions."
        )


elif page == T["analytics"]:
    st.subheader(T["analytics"])
    history = read_history()

    if history.empty:
        st.info("Analyze some images to populate your analytics dashboard.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total scans", len(history))
        b.metric("Unique crops", history["crop"].nunique())
        c.metric("Unique predicted conditions", history["disease"].nunique())

        st.markdown("### Most frequently predicted conditions")
        counts = history["disease"].value_counts()
        st.bar_chart(counts)

        st.markdown("### Scans over time")
        if "date" in history.columns:
            dates = pd.to_datetime(history["date"], errors="coerce")
            daily = dates.dt.date.value_counts().sort_index()
            if not daily.empty:
                st.line_chart(daily)

# ---------- Analytics ----------
        if "score" in history.columns:
            st.metric(
                "Average model output score",
                f"{pd.to_numeric(history['score'], errors='coerce').mean():.2f}%"
            )
            st.caption(
                "Model output scores are not necessarily calibrated probabilities."
            )

        st.markdown("### Recent scan records")
        st.dataframe(history.tail(10), width="stretch", hide_index=True)

# ---------- Disease Library ----------
elif page == T["library"]:
    st.subheader(T["library"])
    try:
        _, class_names = load_model()
        search = st.text_input("Search crop or disease")
        for class_name in class_names:
            crop, disease = decode_class(class_name)
            if search.lower() in f"{crop} {disease}".lower():
                with st.expander(f"{crop} — {disease}"):
                    if "healthy" in disease.lower():
                        st.success("This is a healthy class in the training labels.")
                    else:
                        st.write("General precautions:")
                        for tip in precautions(disease):
                            st.write("• " + tip)
    except Exception as e:
        st.error(str(e))

# ---------- Plant Doctor ----------
elif page == T["doctor"]:
    st.subheader(T["doctor"])
    st.write(
        "This is a rule-based plant-care helper using the information "
        "you enter. It is not a trained conversational diagnosis model."
    )
    crop = st.text_input("Plant or crop name")
    symptoms = st.multiselect(
        "What are you observing?",
        [
            "Yellow leaves", "Brown spots", "White powder",
            "Wilting", "Curling leaves", "Holes in leaves",
            "Rotting areas", "Stunted growth", "No visible symptoms"
        ]
    )
    watering = st.selectbox(
        "How often is the plant watered?",
        ["Daily", "Several times a week", "Weekly", "Not sure"]
    )
    if st.button("Get plant-care suggestions"):
        st.markdown("### Suggestions")
        if not symptoms or "No visible symptoms" in symptoms:
            st.write("Continue routine inspection and suitable watering.")
        if "Yellow leaves" in symptoms:
            st.write("Check watering, drainage and whether older or newer leaves are affected.")
        if "Brown spots" in symptoms:
            st.write("Avoid wetting foliage and inspect whether spots are spreading.")
        if "White powder" in symptoms:
            st.write("Improve airflow and seek local advice about possible mildew.")
        if "Wilting" in symptoms:
            st.write("Check soil moisture and drainage before changing watering.")
        if "Curling leaves" in symptoms:
            st.write("Inspect both sides of leaves for pests and other symptoms.")
        if "Holes in leaves" in symptoms:
            st.write("Inspect for chewing insects and monitor new leaf growth.")
        if "Rotting areas" in symptoms:
            st.write("Avoid excess moisture and seek local advice promptly.")
        if "Stunted growth" in symptoms:
            st.write("Review growing conditions, drainage and nutrient management.")
        st.write(f"Watering entered: {watering}")
        if crop:
            st.write(f"Crop entered: {crop}")
        st.info("These are general suggestions, not a confirmed diagnosis.")

# ---------- Environmental Risk ----------
elif page == T["risk"]:
    st.subheader(T["risk"])
    st.write("Estimate general environmental conditions that may favor some diseases.")
    humidity = st.slider("Relative humidity (%)", 0, 100, 60)
    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    rain = st.radio("Recent or ongoing rain?", ["No", "Yes"], horizontal=True)
    leaf_wetness = st.radio("Are leaves frequently wet?", ["No", "Yes"], horizontal=True)

    if st.button("Estimate environmental risk"):
        score, level, reasons = risk_assessment(
            humidity, rain, temperature, leaf_wetness
        )
        st.metric("Rule-based risk score", f"{score}/100")
        if score >= 60:
            st.warning(level)
        elif score >= 30:
            st.info(level)
        else:
            st.success(level)

        if reasons:
            st.write("Factors contributing to this estimate:")
            for reason in reasons:
                st.write("• " + reason)
        else:
            st.write("No listed risk factors were selected.")
        st.caption(
            "This is a simple educational estimate, not a crop-specific forecast. "
            "It does not use live weather data."
        )

# ---------- Crop Disease Risk Map ----------
elif page == T["map"]:
    st.subheader("🌍 Crop Disease Risk Map")
    st.write(
        "Search for a farm town or city to view current weather and a "
        "rule-based estimate of environmental conditions that may favor disease."
    )
    map_city = st.text_input(
        "Enter farm city or town", value="Bengaluru", key="crop_risk_map_city"
    )

    if st.button("Check location risk", type="primary", key="check_location_risk"):
        if not map_city.strip():
            st.warning("Enter a city or town name first.")
        else:
            try:
                with st.spinner("Finding location and retrieving weather..."):
                    geo_response = requests.get(
                        "https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": map_city.strip(), "count": 1, "language": "en"},
                        timeout=15
                    )
                    geo_response.raise_for_status()
                    places = geo_response.json().get("results", [])

                    if not places:
                        st.warning("Location not found. Try a nearby larger town or city.")
                    else:
                        place = places[0]
                        weather_response = requests.get(
                            "https://api.open-meteo.com/v1/forecast",
                            params={
                                "latitude": place["latitude"],
                                "longitude": place["longitude"],
                                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                                "daily": "precipitation_probability_max,temperature_2m_max,temperature_2m_min",
                                "forecast_days": 7,
                                "timezone": "auto"
                            },
                            timeout=15
                        )
                        weather_response.raise_for_status()
                        weather_data = weather_response.json()
                        current = weather_data["current"]
                        humidity_now = float(current["relative_humidity_2m"])
                        temperature_now = float(current["temperature_2m"])
                        rainfall_now = float(current["precipitation"])

                        risk_score, risk_level, risk_reasons = risk_assessment(
                            humidity_now,
                            "Yes" if rainfall_now > 0 else "No",
                            temperature_now,
                            "Unknown"
                        )

                        st.markdown(
                            f"### 📍 {place.get('name', map_city)}, "
                            f"{place.get('admin1', '')}, {place.get('country', '')}"
                        )
                        wx1, wx2, wx3 = st.columns(3)
                        wx1.metric("Temperature", f"{temperature_now:.1f} °C")
                        wx2.metric("Humidity", f"{humidity_now:.0f}%")
                        wx3.metric("Current precipitation", f"{rainfall_now:.1f} mm")

                        st.markdown("#### Estimated environmental disease-favouring conditions")
                        st.metric("Weather risk estimate", f"{risk_score}/100")
                        st.progress(risk_score / 100)
                        st.write(f"**Category:** {risk_level}")
                        if risk_score >= 60:
                            st.warning("Monitor crops closely and inspect leaves regularly.")
                        elif risk_score >= 30:
                            st.info("Continue routine crop monitoring and avoid unnecessary leaf wetness.")
                        else:
                            st.success("Fewer of the selected weather risk factors are present now.")
                        if risk_reasons:
                            st.write("**Contributing factors:**")
                            for reason in risk_reasons:
                                st.write("- " + reason)

                        st.map(pd.DataFrame({
                            "lat": [place["latitude"]],
                            "lon": [place["longitude"]]
                        }))

                        daily = weather_data.get("daily", {})
                        if daily.get("time"):
                            forecast_rows = []
                            for i, day in enumerate(daily["time"]):
                                forecast_rows.append({
                                    "Date": day,
                                    "Minimum °C": daily["temperature_2m_min"][i],
                                    "Maximum °C": daily["temperature_2m_max"][i],
                                    "Rain chance %": daily["precipitation_probability_max"][i]
                                })
                            st.markdown("#### 7-Day Local Weather Outlook")
                            st.dataframe(forecast_rows, hide_index=True, width="stretch")

                        st.caption(
                            "The map pin marks the searched town only. The score is a simple "
                            "weather-based educational estimate, not a disease-outbreak map, "
                            "confirmed diagnosis, or crop-specific forecast. Leaf wetness is "
                            "not measured by this weather service."
                        )
            except requests.RequestException:
                st.error("Weather service could not be reached. Check your internet and retry.")
            except (KeyError, TypeError, ValueError, IndexError):
                st.error("Weather data was incomplete or could not be processed. Please retry.")

# ---------- About ----------
elif page == T["about"]:
    st.subheader(T["about"])
    st.write("""
    **PlantCare AI** is an educational plant disease detection project.

    The project uses a pretrained MobileNetV2 transfer-learning model
    to classify leaf images into 38 supported classes. It achieved 90.41%
    accuracy on the project held-out test set of 8,146 images.

    **Technology used**
    - Python
    - TensorFlow / Keras
    - Streamlit
    - NumPy and Pandas
    - Pillow
    - ReportLab for PDF reports

    **Important limitations**
    - Predictions are limited to the model's 38 training classes.
    - Model output scores are not guaranteed to be calibrated probabilities.
    - Severity and health scores are based on user-entered estimates.
    - Environmental risk is rule-based and not a validated forecast.
    - Plant Doctor suggestions are general guidance, not professional diagnosis.
    - The camera feature captures a still image, not continuous live video.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("PlantCare AI | Educational project")
st.caption(
    "Disclaimer: AI predictions and care suggestions are estimates. "
    "Confirm important crop-health decisions with a qualified local agricultural expert."
)
