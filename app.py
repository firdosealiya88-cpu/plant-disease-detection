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
MODEL_PATH = ROOT / "models" / "color_cnn_v2.keras"
CLASS_PATH = ROOT / "outputs" / "cnn" / "class_names_v2.json"
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
        "risk": "Environmental Risk", "map": "Plant Health Map",
        "about": "About Project", "upload": "Upload plant leaf images",
        "predict": "Analyze image", "healthy": "Healthy",
        "diseased": "Possible disease detected"
    },
    "Hindi": {
        "home": "होम", "detect": "रोग पहचान",
        "history": "पहचान इतिहास", "analytics": "विश्लेषण डैशबोर्ड",
        "library": "रोग पुस्तकालय", "doctor": "प्लांट डॉक्टर",
        "risk": "पर्यावरणीय जोखिम", "map": "पौधों का स्वास्थ्य मानचित्र",
        "about": "परियोजना के बारे में", "upload": "पत्तियों की तस्वीरें अपलोड करें",
        "predict": "तस्वीर का विश्लेषण करें", "healthy": "स्वस्थ",
        "diseased": "संभावित रोग का पता चला"
    },
    "Kannada": {
        "home": "ಮುಖಪುಟ", "detect": "ರೋಗ ಪತ್ತೆ",
        "history": "ಪತ್ತೆ ಇತಿಹಾಸ", "analytics": "ವಿಶ್ಲೇಷಣೆ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "library": "ರೋಗ ಗ್ರಂಥಾಲಯ", "doctor": "ಪ್ಲಾಂಟ್ ಡಾಕ್ಟರ್",
        "risk": "ಪರಿಸರ ಅಪಾಯ", "map": "ಸಸ್ಯ ಆರೋಗ್ಯ ನಕ್ಷೆ",
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
    array = np.asarray(resized, dtype=np.float32) / 255.0
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
    T["home"], T["detect"], T["history"], T["analytics"],
    T["library"], T["doctor"], T["risk"], T["map"], T["about"]
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
    f.metric("CNN supported classes", 38)

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


    st.divider()
    st.subheader("Live Weather Monitoring")
    st.caption("Weather information fetched from Open-Meteo. Enter your farm or city location.")

    weather_col, refresh_col = st.columns([3, 1])
    with weather_col:
        weather_city = st.text_input(
            "Enter city or location",
            value="Bengaluru",
            key="live_weather_city"
        )
    with refresh_col:
        st.write("")
        refresh_weather = st.button(
            "Refresh weather",
            key="refresh_live_weather",
            width="stretch"
        )

    if refresh_weather:
        get_live_weather.clear()

    if weather_city.strip():
        try:
            with st.spinner("Fetching live weather..."):
                live_data = get_live_weather(weather_city.strip())

            if live_data is None:
                st.warning("Location not found. Check the city name and try again.")
            else:
                place = live_data["place"]
                current = live_data["weather"]["current"]
                daily = live_data["weather"]["daily"]

                st.caption(
                    f"Location: {place.get('name', weather_city)}, "
                    f"{place.get('admin1', '')}, {place.get('country', '')} "
                    f"| Updated: {current.get('time', 'N/A')}"
                )

                st.markdown(
                    f"**{weather_description(current.get('weather_code'))}**"
                )

                weather_metrics = st.columns(4)
                weather_metrics[0].metric(
                    "Temperature",
                    f"{current.get('temperature_2m', 'N/A')} °C"
                )
                weather_metrics[1].metric(
                    "Humidity",
                    f"{current.get('relative_humidity_2m', 'N/A')}%"
                )
                weather_metrics[2].metric(
                    "Rainfall",
                    f"{current.get('precipitation', 'N/A')} mm"
                )
                weather_metrics[3].metric(
                    "Wind speed",
                    f"{current.get('wind_speed_10m', 'N/A')} km/h"
                )

                st.markdown("#### ?? Weather-Based Environmental Risk Alert")

                try:
                    weather_humidity = float(current["relative_humidity_2m"])
                    weather_rainfall = float(current["precipitation"])
                    weather_temperature = float(current["temperature_2m"])

                    if not np.isfinite([
                        weather_humidity,
                        weather_rainfall,
                        weather_temperature
                    ]).all():
                        raise ValueError("Invalid weather reading")

                    rain_status = "Yes" if weather_rainfall > 0 else "No"

                    weather_score, weather_level, weather_reasons = risk_assessment(
                        weather_humidity,
                        rain_status,
                        weather_temperature,
                        "Unknown"
                    )

                    alert_cols = st.columns(2)
                    alert_cols[0].metric(
                        "Environmental risk score",
                        f"{weather_score}/100"
                    )
                    alert_cols[1].metric(
                        "Estimated risk level",
                        weather_level
                    )

                    st.progress(weather_score / 100)

                    if weather_score >= 60:
                        st.warning(
                            "Higher environmental risk: inspect plants regularly "
                            "for unusual spots, discoloration, or leaf changes."
                        )
                    elif weather_score >= 30:
                        st.info(
                            "Moderate environmental risk: monitor leaves and "
                            "avoid unnecessary moisture on foliage."
                        )
                    else:
                        st.success(
                            "Lower estimated environmental risk from the "
                            "weather factors currently measured."
                        )

                    if weather_reasons:
                        st.write("Contributing factors:")
                        for reason in weather_reasons:
                            st.write(f"- {reason}")

                    st.caption(
                        "This is a rule-based environmental estimate, not a "
                        "disease diagnosis or validated disease forecast. "
                        "Live weather does not measure actual leaf wetness, "
                        "so that factor is excluded from this score."
                    )

                except (KeyError, TypeError, ValueError):
                    st.info(
                        "A weather-based risk alert is unavailable because "
                        "one or more readings are missing or invalid."
                    )

                extra_metrics = st.columns(2)
                extra_metrics[0].metric(
                    "Feels like",
                    f"{current.get('apparent_temperature', 'N/A')} °C"
                )
                extra_metrics[1].metric(
                    "Cloud cover",
                    f"{current.get('cloud_cover', 'N/A')}%"
                )

                st.markdown("#### 3-Day Weather Forecast")
                forecast_rows = []
                for i, day in enumerate(daily["time"]):
                    forecast_rows.append({
                        "Date": day,
                        "Condition": weather_description(daily["weather_code"][i]),
                        "Min temperature (°C)": daily["temperature_2m_min"][i],
                        "Max temperature (°C)": daily["temperature_2m_max"][i],
                        "Rain chance (%)": daily["precipitation_probability_max"][i]
                    })

                st.dataframe(
                    forecast_rows,
                    hide_index=True,
                    width="stretch"
                )

                st.caption(
                    "Weather is environmental information, not a plant-disease "
                    "diagnosis. Actual leaf wetness and field conditions may differ."
                )

        except requests.RequestException as error:
            st.error(
                "Could not fetch live weather. Check your internet connection "
                "and try Refresh weather again."
            )
        except (KeyError, ValueError, TypeError, IndexError):
            st.error("The weather service returned unexpected data. Please refresh.")
    else:
        st.info("Enter a city name to view its weather.")


# ---------- Detection ----------
elif page == T["detect"]:
    st.subheader(T["detect"])
    st.caption("Your existing trained CNN is used. No retraining is required.")

    uploaded_files = st.file_uploader(
        T["upload"],
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    camera_file = st.camera_input("Or take a leaf photo")

    severity = st.slider(
        "Estimated percentage of leaf area affected (your visual estimate)",
        min_value=0, max_value=100, value=0, step=5
    )

    if mode == "Expert Mode":
        st.caption(
            "Expert mode: review image quality, top-three model outputs "
            "and the limits of the prediction."
        )

    files_to_process = list(uploaded_files or [])
    if camera_file is not None:
        files_to_process.append(camera_file)

    if files_to_process and st.button(T["predict"], type="primary"):
        try:
            model, class_names = load_model()
            for file in files_to_process:
                image = Image.open(file)
                original, batch = prepare_image(image)
                quality = quality_check(original)

                with st.spinner(f"Analyzing {file.name}..."):
                    predictions = model.predict(batch, verbose=0)[0]

                if len(predictions) != len(class_names):
                    st.error("Model output and class list do not match.")
                    continue

                top_indices = np.argsort(predictions)[::-1][:3]
                best_index = int(top_indices[0])
                predicted_class = class_names[best_index]
                crop, disease = decode_class(predicted_class)
                score = float(predictions[best_index]) * 100
                is_healthy = "healthy" in disease.lower()

                if severity <= 10:
                    severity_label = "Low user-estimated affected area"
                elif severity <= 40:
                    severity_label = "Moderate user-estimated affected area"
                else:
                    severity_label = "High user-estimated affected area"

                health_score = max(0, 100 - severity)
                tips = precautions(disease)
                date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                result = {
                    "date": date_now,
                    "image": file.name,
                    "crop": crop,
                    "disease": disease,
                    "score": score,
                    "severity": f"{severity}% - {severity_label}",
                    "precautions": tips
                }

                save_history({
                    "date": date_now,
                    "image": file.name,
                    "crop": crop,
                    "disease": disease,
                    "score": round(score, 2),
                    "severity_percent_user_estimate": severity,
                    "health_score_user_estimate": health_score
                })

                st.markdown("---")
                st.markdown(f"### Results: {file.name}")
                left, right = st.columns([1, 1])
                with left:
                    st.image(original, caption="Uploaded leaf image", width="stretch")
                with right:
                    st.metric("Predicted crop", crop)
                    st.metric("Predicted condition", disease)
                    st.metric("Model output score", f"{score:.2f}%")
                    st.metric("User-estimated plant health score", f"{health_score}/100")
                    st.caption(
                        "The health score is a simple calculation from your "
                        "entered affected-area estimate, not a CNN diagnosis."
                    )

                if is_healthy:
                    st.success("The model's top prediction is a healthy class.")
                else:
                    st.warning("The model's top prediction suggests a possible disease.")

                st.markdown("#### Image quality check")
                st.write(
                    f"Resolution: {quality['width']} × {quality['height']} | "
                    f"Brightness: {quality['brightness']} | "
                    f"Sharpness indicator: {quality['sharpness']}"
                )
                if quality["notes"]:
                    for note in quality["notes"]:
                        st.warning(note)
                else:
                    st.success("No basic image-quality warning detected.")

                st.markdown("#### Top 3 model outputs")
                top_data = []
                for idx in top_indices:
                    c, d = decode_class(class_names[int(idx)])
                    top_data.append({
                        "Crop": c,
                        "Condition": d,
                        "Model output": f"{float(predictions[int(idx)]) * 100:.2f}%"
                    })
                st.dataframe(pd.DataFrame(top_data), hide_index=True, width="stretch")

                st.markdown("#### Suggested precautions")
                for tip in tips:
                    st.write("• " + tip)

                st.download_button(
                    "Download PDF report",
                    data=create_pdf(result),
                    file_name="plantcare_report.pdf",
                    mime="application/pdf",
                    key=f"pdf_{file.name}_{date_now}"
                )

        except Exception as e:
            st.error(f"Could not analyze the image: {e}")

# ---------- History ----------
elif page == T["history"]:
    st.subheader(T["history"])
    history = read_history()

    if history.empty:
        st.info("No scans have been saved yet. Analyze an image first.")
    else:
        st.dataframe(history, width="stretch", hide_index=True)
        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scan history CSV",
            data=csv_data,
            file_name="plantcare_scan_history.csv",
            mime="text/csv"
        )
        if st.button("Clear all scan history"):
            HISTORY_PATH.unlink(missing_ok=True)
            st.success("History cleared. Refresh the page.")
            st.rerun()

# ---------- Analytics ----------
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

# ---------- Map ----------
elif page == T["map"]:
    st.subheader(T["map"])
    st.write(
        "Demonstration map: enter approximate farm or garden coordinates "
        "to display a marker. Do not enter a private home address."
    )
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.9716, format="%.5f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.5946, format="%.5f")
    label = st.text_input("Location label", value="My garden")

    if st.button("Show location on map"):
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
        st.success(f"Showing map marker for: {label}")
    st.caption(
        "This map displays only the coordinates you enter. It does not show "
        "live disease outbreaks or verified regional disease risk."
    )

# ---------- About ----------
elif page == T["about"]:
    st.subheader(T["about"])
    st.write("""
    **PlantCare AI** is an educational plant disease detection project.

    The project uses a previously trained convolutional neural network (CNN)
    to classify leaf images into 38 supported classes.

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
