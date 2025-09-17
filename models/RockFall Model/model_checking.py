import pandas as pd
import joblib
import os
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ===============================
# Step 1: Define File Paths
# ===============================
# IMPORTANT: Replace these paths with the actual location of your files
model_path = r"D:\SIH_2025\coal\kusmunda\rockfall_model.pkl"
scaler_path = r"D:\SIH_2025\coal\kusmunda\scaler.pkl"
feature_columns_path = r"D:\SIH_2025\coal\kusmunda\feature_columns.pkl"

if __name__ == "__main__":
    # ===============================
    # Step 2: Load Saved Model + Preprocessors
    # ===============================
    print("--- STARTING DIRECT ROCKFALL PREDICTION ---")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(feature_columns_path)
        print("Successfully loaded model and preprocessors.")
    except FileNotFoundError as e:
        print("--- ERROR ---")
        print(f"A required file was not found: {e}")
        print("Please ensure your model, scaler, and feature_columns files are in the specified directory.")
        exit()

    # ===============================
    # Step 3: Define Manual Input
    # ===============================
    user_input = {
        'Sector': 'high_risk',
        'Avg_Slope_Deg': 34.613,
        'Avg_Roughness_dry': 3.39,
        'Rainfall_mm': 80,
        'Temperature_C': 29,
        'Windspeed_mps': 0,
        'Displacement': 0.466,
        'Strain': 0.0045,
        'Pore_Pressure': 0.69,
        'Shear_Stress': 15045
    }

    # ===============================
    # Step 4: Preprocess Input
    # ===============================
    # Convert input dictionary to a pandas DataFrame
    input_df = pd.DataFrame([user_input])

    # One-Hot Encode the 'Sector' column
    input_df = pd.get_dummies(input_df, columns=['Sector'], drop_first=True)

    # Align columns with the training data to ensure all features are present
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale the numerical features
    numerical_cols = [col for col in user_input.keys() if col != 'Sector']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # ===============================
    # Step 5: Prediction
    # ===============================
    predictions = model.predict(input_df)[0]
    safety_factor = predictions[0]
    rockfall_prob = predictions[1]
    
    # ===============================
    # Step 6: Print Results
    # ===============================
    print("\nInputs provided:")
    for key, val in user_input.items():
        print(f"  - {key}: {val}")
    
    print("\n--- FINAL PREDICTION RESULTS ---")
    print(f">>> Predicted Safety Factor: {safety_factor:.4f}")
    print(f">>> Predicted Rockfall Probability: {rockfall_prob:.4f}")
    print("---------------------------------")
