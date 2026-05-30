import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset
csv_path = Path(__file__).with_name("Crop_recommendation.csv")
data = pd.read_csv(csv_path)

# Features and Target
X = data.drop("label", axis=1)
y = data["label"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🌱 Crop Recommendation System")

st.write("Enter the soil and weather details below:")

# User Inputs
N = st.number_input("Nitrogen (N)", 0, 150, 50)
P = st.number_input("Phosphorus (P)", 0, 150, 50)
K = st.number_input("Potassium (K)", 0, 150, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

# Prediction
if st.button("Predict Crop"):

    input_data = pd.DataFrame([
        [N, P, K, temperature, humidity, ph, rainfall]
    ], columns=X.columns)

    prediction = model.predict(input_data)
    crop_name = le.inverse_transform(prediction)

    st.success(f"Recommended Crop: {crop_name[0]}")
    