import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
import os

# --- App Configuration ---
st.set_page_config(page_title="Outbreak Predictor", layout="wide")
st.title("📈 Real-Time Disease Outbreak Predictor")
st.markdown("Select a location and enter today's data to forecast the next day's new confirmed cases.")

# --- Caching: Load data only once ---
@st.cache_data
def load_data():
    """Loads data and assets from the correct project folders."""
    
    # ⭐️ DEFINE YOUR PROJECT FOLDER PATHS
    MODELS_DIR = r"C:\Users\Deepak\OneDrive\Desktop\ObjectRegon\Deepak_AML_Project_Code\models"
    DATA_DIR = r"C:\Users\Deepak\OneDrive\Desktop\ObjectRegon\Deepak_AML_Project_Code\data"

    # Construct full paths to the required files
    data_path = os.path.join(DATA_DIR, "merged_dataset_with_geo.parquet")
    loc2idx_path = os.path.join(MODELS_DIR, "loc2idx_no_lag.pkl")
    feature_cols_path = os.path.join(MODELS_DIR, "feature_cols_no_lag.pkl")
    
    # Load all files from their specific paths
    df = pd.read_parquet(data_path)
    loc2idx = joblib.load(loc2idx_path)
    feature_cols = joblib.load(feature_cols_path)
    
    locations = list(loc2idx.keys())
    
    # Pre-calculate features
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['location_key', 'date'], inplace=True)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df, locations, feature_cols

# --- Load Data ---
df, locations, feature_cols = load_data()

# --- UI Layout ---
# (The rest of the UI layout and prediction logic remains the same)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Select Location")
    selected_location = st.selectbox(
        "Select a Location",
        options=locations,
        index=locations.index("US_AL") if "US_AL" in locations else 0
    )

    st.subheader("2. Enter Today's Data")
    user_new_deceased = st.number_input("Today's New Deceased", min_value=0, step=1)
    user_hospitalized = st.number_input("Current Hospitalized Patients", min_value=0, step=1)
    user_icu = st.number_input("Current ICU Patients", min_value=0, step=1)
    user_stringency = st.slider("Government Stringency Index (0-100)", 0, 100, 50)
    
    historical_df = df[df['location_key'] == selected_location].tail(29)

with col2:
    st.subheader("3. Get Forecast")
    if st.button("▶️ Run Prediction"):
        if len(historical_df) < 29:
            st.error(f"Error: Not enough historical data for '{selected_location}'.")
        else:
            with st.spinner("Building sequence and running prediction..."):
                last_known_row = historical_df.iloc[-1]
                today_features = {
                    "new_deceased": user_new_deceased,
                    "current_hospitalized_patients": user_hospitalized,
                    "current_intensive_care_patients": user_icu,
                    "cumulative_persons_vaccinated": last_known_row['cumulative_persons_vaccinated'],
                    "cumulative_persons_fully_vaccinated": last_known_row['cumulative_persons_fully_vaccinated'],
                    "average_temperature_celsius": last_known_row['average_temperature_celsius'],
                    "relative_humidity": last_known_row['relative_humidity'],
                    "rainfall_mm": last_known_row['rainfall_mm'],
                    "stringency_index": float(user_stringency),
                    "day_of_week": pd.Timestamp.now().dayofweek,
                    "month": pd.Timestamp.now().month
                }
                
                historical_sequence = historical_df[feature_cols].values.tolist()
                today_sequence_values = [today_features[col] for col in feature_cols]
                full_sequence_list = historical_sequence + [today_sequence_values]
                
                api_input = {
                    "sequence": [{"features": [float(f) for f in feature_set]} for feature_set in full_sequence_list],
                    "location_key": selected_location
                }

                api_url = "http://127.0.0.1:8000/predict"
                try:
                    response = requests.post(api_url, json=api_input)
                    if response.status_code == 200:
                        prediction = response.json()
                        predicted_cases = prediction.get("predicted_new_cases", "N/A")
                        st.success("Forecast complete!")
                        st.metric(
                            label=f"Predicted New Cases for Tomorrow in {selected_location}",
                            value=f"{predicted_cases:,.0f}"
                        )
                    else:
                        st.error(f"Error from API: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Could not connect to the API.")