import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from huggingface_hub import hf_hub_download

# ==========================================
# 1. SETUP & PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="🍃",
    layout="wide"
)

st.title("🍃 Air Quality Prediction & Analysis System")
st.markdown("Enter sensor data below to predict **PM2.5 Levels** and **Air Quality Classification**.")

# ==========================================
# 2. LOAD MODELS FROM HUGGING FACE HUB
# ==========================================
@st.cache_resource
def load_artifacts():
    repo_id = "hridayeshdebsarma6/Air-Quality-Prediction-Analysis-System"
    
    try:
        st.info("Downloading models from Hugging Face Hub... (This may take a moment)")
        
        # Download files locally from the hub
        reg_path = hf_hub_download(repo_id=repo_id, filename="regression_models.joblib", repo_type="model")
        class_path = hf_hub_download(repo_id=repo_id, filename="classification_models.joblib", repo_type="model")
        scaler_path = hf_hub_download(repo_id=repo_id, filename="scaler.joblib", repo_type="model")
        
        # Load the models using joblib
        reg_models = joblib.load(reg_path)
        class_models = joblib.load(class_path)
        scaler = joblib.load(scaler_path)
        
        st.success("Models loaded successfully!")
        return reg_models, class_models, scaler

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

reg_models, class_models, scaler = load_artifacts()

if reg_models is None:
    st.stop()

# ==========================================
# 3. SIDEBAR - MODEL SELECTION
# ==========================================
st.sidebar.header("⚙️ Model Settings")

task_type = st.sidebar.radio("Select Task", ["Regression (Predict PM2.5 Value)", "Classification (Good vs Bad Air)"])

if task_type.startswith("Regression"):
    model_name = st.sidebar.selectbox("Choose Regression Model", list(reg_models.keys()))
    active_model = reg_models[model_name]
else:
    model_name = st.sidebar.selectbox("Choose Classification Model", list(class_models.keys()))
    active_model = class_models[model_name]

st.sidebar.markdown("---")
st.sidebar.info(f"Using: **{model_name}**")

# ==========================================
# 4. USER INPUT FORM
# ==========================================
with st.form("prediction_form"):
    st.subheader("📝 Input Environmental Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        temp = st.number_input("Temperature (°F)", value=70.0)
        pressure = st.number_input("Pressure (hPa)", value=1000.0)
        
    with col2:
        pm1 = st.number_input("PM 1.0", value=10.0)
        pm10 = st.number_input("PM 10.0", value=20.0)
        voc = st.number_input("VOC (Volatile Organic Compounds)", value=100.0)
        
    with col3:
        lat = st.number_input("Latitude", value=37.77, format="%.4f")
        lon = st.number_input("Longitude", value=-122.41, format="%.4f")
        date_time = st.date_input("Date", datetime.date.today())
        time_input = st.time_input("Time", datetime.datetime.now().time())

    # Submit Button
    submitted = st.form_submit_button("🚀 Run Prediction")

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if submitted:
    # --- A. Preprocessing (Replicating Training Logic) ---
    full_datetime = datetime.datetime.combine(date_time, time_input)
    
    hour = full_datetime.hour
    month = full_datetime.month
    day_of_week = full_datetime.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    lat_round = round(lat, 2)
    lon_round = round(lon, 2)

    input_data = pd.DataFrame({
        'HUMIDITY': [humidity],
        'TEMPERATURE': [temp],
        'PRESSURE': [pressure],
        'PM1': [pm1],
        'PM10': [pm10],
        'VOC': [voc],
        'HOUR': [hour],
        'MONTH': [month],
        'DAY_OF_WEEK': [day_of_week],
        'IS_WEEKEND': [is_weekend],
        'LAT_ROUND': [lat_round],
        'LON_ROUND': [lon_round]
    })

    # --- B. Scaling ---
    input_scaled = scaler.transform(input_data)

    # --- C. Prediction ---
    prediction = active_model.predict(input_scaled)

    # --- D. Display Results ---
    st.markdown("---")
    
    if task_type.startswith("Regression"):
        pred_value = prediction[0]
        st.success(f"### Predicted PM2.5 Level: {pred_value:.2f}")
        
        if pred_value <= 12.0:
            st.info("Air Quality is likely **Good** 🟢")
        elif pred_value <= 35.4:
            st.warning("Air Quality is likely **Moderate** 🟡")
        else:
            st.error("Air Quality is likely **Unhealthy** 🔴")
            
    else:
        pred_class = prediction[0]
        if pred_class == 0:
            st.success("### Predicted Class: Good Air Quality 🟢")
        else:
            st.error("### Predicted Class: Poor Air Quality 🔴")