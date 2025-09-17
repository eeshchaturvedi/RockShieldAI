# ===============================
# Step 1: Import Libraries
# ===============================
import joblib
import pandas as pd

# ===============================
# Step 2: Load Saved Model + Encoder + Scaler (if used)
# ===============================
model = joblib.load(r"D:\SIH_2025\coal\kusmunda\strain models\strain_model_lgbm.pkl")   # replace with your model path
encoder = joblib.load(r"D:\SIH_2025\coal\kusmunda\strain models\label_encoder.pkl")  # LabelEncoder for Sector
scaler = joblib.load(r"D:\SIH_2025\coal\kusmunda\strain models\scaler_strain.pkl")  # if you used StandardScaler

# ===============================
# Step 3: Define Manual Input
# ===============================
manual_input = {
    "Sector": "sonpuri_3",         # example sector name
    "Avg_Slope_Deg": 38.5,
    "Avg_Roughness_dry": 0.72,
    "Rainfall_mm": 120.0,
    "Temperature_C": 32.4,
    "Windspeed_mps": 3.1,
    "Displacement": 0.045,
    "Pore_Pressure": 1.12,
    "Shear_Stress": 5.8
}

# ===============================
# Step 4: Preprocess Input
# ===============================
# Convert to DataFrame
input_df = pd.DataFrame([manual_input])

# Encode Sector
input_df["Sector"] = encoder.transform(input_df["Sector"])

# Scale features (if you scaled during training)
X_scaled = scaler.transform(input_df)

# ===============================
# Step 5: Prediction
# ===============================
predicted_strain = model.predict(X_scaled)[0]
print("📌 Predicted Strain:", predicted_strain)
