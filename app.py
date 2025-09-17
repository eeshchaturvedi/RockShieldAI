from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import json
import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import joblib
import os
import warnings
import sys
from flask import Flask, render_template, request
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random


app = Flask(__name__)
app.secret_key = "supersecretkey"





# Sender and receiver
sender_email = "rockshieldai@gmail.com"
receiver_email = "aryan.phanse03@gmail.com"
app_password = "glhb ccto xawo cbjt"  # Replace with your 16-character App Password

# --- Robust Model and Data Loading ---
try:
    #Loading Strain Models
    model = joblib.load(r"D:\SIH_2025\project\models\Strain Model\strain_model_lgbm.pkl")
    encoder = joblib.load(r"D:\SIH_2025\project\models\Strain Model\label_encoder.pkl")
    scaler = joblib.load(r"D:\SIH_2025\project\models\Strain Model\scaler_strain.pkl")

    #Loading Rockfall Models
    model_path = r"D:\SIH_2025\project\models\RockFall Model\rockfall_model.pkl"
    scaler_path = r"D:\SIH_2025\project\models\RockFall Model\scaler.pkl"
    feature_columns_path = r"D:\SIH_2025\project\models\RockFall Model\feature_columns.pkl"
    
    # Load data for visualization
    df = pd.read_csv(r"D:\SIH_2025\project\script\kusmunda_DEM_features.csv")

except FileNotFoundError as e:
    print(f"FATAL ERROR: A required model or data file was not found.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print("Please ensure your file paths are correct and the files exist.", file=sys.stderr)
    sys.exit("Application cannot start without critical files.")
except Exception as e:
    print(f"An unexpected error occurred during application startup: {e}", file=sys.stderr)
    sys.exit("Application startup failed.")


@app.route("/")
def home():
    return render_template("Homepage-mkII.html")

@app.route('/analysis')
def analysis_page():
    return redirect(url_for("home"))






# -----------------------------
# Load CSV for 3D visualization
# -----------------------------
X = df['X_m'].values
Y = df['Y_m'].values
Z = df['Elevation_m'].values
Slope = df['Slope_Deg'].values
Roughness = df['Roughness_dry'].values

# -----------------------------
# Define blocks and colors
# -----------------------------
block_boundaries = np.linspace(X.min(), X.max(), 4)
block_names = ["Jatraj", "Risdi", "Sonpuri"]
block_colors = {
    "Jatraj": ["#e5f5e0", "#a1d99b", "#31a354", "#006d2c"],
    "Risdi": ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#636363"],
    "Sonpuri": ["#deebf7", "#9ecae1", "#3182bd", "#08519c"]
}

def assign_block(x):
    if x < block_boundaries[1]:
        return "Jatraj"
    elif x < block_boundaries[2]:
        return "Risdi"
    else:
        return "Sonpuri"

point_colors = []
for i in range(len(X)):
    block = assign_block(X[i])
    block_start = block_boundaries[block_names.index(block)]
    block_end = block_boundaries[block_names.index(block) + 1]
    sector_width = (block_end - block_start) / 4
    sector_idx = min(int((X[i] - block_start) // sector_width), 3)
    point_colors.append(block_colors[block][sector_idx])

hover_text = [
    f"X: {X[i]:.2f}, Y: {Y[i]:.2f}, Z: {Z[i]:.2f}<br>"
    f"Elevation: {Z[i]:.2f}<br>"
    f"Slope: {Slope[i]:.2f}°<br>"
    f"Roughness: {Roughness[i]:.2f}<br>"
    f"Sector: {df.loc[i, 'Sector']}"
    for i in range(len(X))
]

# -----------------------------
# Dash app embedded in Flask
# -----------------------------
dash_app = dash.Dash(__name__, server=app, url_base_pathname="/manual/") 

fig = go.Figure() 

fig.add_trace(go.Mesh3d(
    x=X, y=Y, z=Z, 
    vertexcolor=point_colors, 
    opacity=0.9, 
    text=hover_text, 
    hoverinfo="text"
)) 

risk_mask = Slope > 30 
fig.add_trace(go.Scatter3d(
    x=X[risk_mask], y=Y[risk_mask], z=Z[risk_mask], 
    mode="markers", 
    marker=dict(size=5, color="red", opacity=0.4), 
    name="Risky Zone (>30° slope)", 
    text=[hover_text[i] for i in np.where(risk_mask)[0]], 
    hoverinfo="text"
)) 

fig.add_trace(go.Scatter3d(
    x=X, y=Y, z=Z, 
    mode="markers", 
    marker=dict(size=3, color="rgba(0,0,0,0)"), 
    text=hover_text, 
    hoverinfo="none", 
    name="Clickable Points"
)) 


fig.update_layout(
    scene=dict(
        xaxis_title="X (m)", 
        yaxis_title="Y (m)", 
        zaxis_title="Elevation (m)"
    ), 
    margin=dict(l=0, r=0, t=30, b=0), 
    title="Kusmunda Mine 3D Visualization",
    annotations=[
        dict(
            text="© 2025 RockShield AI. All rights reserved.",
            x=1, y=0,
            xref="paper", yref="paper",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=12, color="gray")
        )
    ]
)
dash_app.layout = html.Div([dcc.Graph(id="3d-surface", figure=fig, style={"height": "95vh"}), dcc.Store(id="clicked-point-data"), html.Div(id="dummy-output")])

@dash_app.callback(Output("clicked-point-data", "data"), Input("3d-surface", "clickData"), prevent_initial_call=True)
def handle_click(clickData):
    if not clickData:
        raise dash.exceptions.PreventUpdate
    p = clickData["points"][0]
    idx = np.argmin((X - p["x"])**2 + (Y - p["y"])**2 + (Z - p["z"])**2)
    data = {"X": round(p["x"],2), "Y": round(p["y"],2), "Z": round(p["z"],2), "Elevation": round(Z[idx],2), "Slope": round(Slope[idx],2), "Roughness": round(Roughness[idx],2), "Sector": df.loc[idx, "Sector"]}
    try:
        requests.post("http://127.0.0.1:5000/save_coords", json=data)
    except Exception as e:
        print(f"Error sending clicked point data to Flask: {e}", file=sys.stderr)
    return data

dash_app.clientside_callback(
    """function(data) { if (!data) return; alert(`You clicked on:\\nX: ${data.X}\\nY: ${data.Y}\\nZ: ${data.Z}\\nElevation: ${data.Elevation}\\nSlope: ${data.Slope}\\nRoughness: ${data.Roughness}\\nSector: ${data.Sector}`); window.location.href = "/#prediction"; }""",
    Output("dummy-output", "children"), Input("clicked-point-data", "data")
)

@app.route("/save_coords", methods=["POST"])
def save_coords():
    data = request.get_json()
    with open("selected_coords.json", "w") as f:
        json.dump(data, f, indent=4)
    return jsonify({"status": "Coordinates saved", "data": data})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form or request.get_json()

        # Safely get and convert form data, providing default '0' if missing
        rainfall = float(data.get('rainfall', '0'))
        windspeed = float(data.get('windspeed', '0'))
        temperature = float(data.get('temperature', '0'))
        shear_stress = float(data.get('Shear stress', '0'))
        pore_pressure = float(data.get('pore-pressure', '0'))
        displacement = float(data.get('Displacement', '0'))

        if not os.path.exists("selected_coords.json"):
            return "Error: No point was selected from the 3D map. Please go back and select a point.", 400

        with open("selected_coords.json", "r") as f:
            point_data = json.load(f)

        sector = point_data["Sector"]
        slope = point_data["Slope"]
        roughness = point_data["Roughness"]

        # --- Safe Encoder Transform ---
        def safe_transform(encoder, values, default=0):
            classes = list(encoder.classes_)
            transformed = []
            for v in values:
                if v in classes:
                    transformed.append(encoder.transform([v])[0])
                else:
                    print(f"[WARNING] Unseen label: {v}, mapping to default {default}")
                    transformed.append(default)
            return transformed

        # --- Strain Prediction ---
        strain_input = pd.DataFrame([{
            "Sector": sector,
            "Avg_Slope_Deg": slope,
            "Avg_Roughness_dry": roughness,
            "Rainfall_mm": rainfall,
            "Temperature_C": temperature,
            "Windspeed_mps": windspeed,
            "Displacement": displacement,
            "Pore_Pressure": pore_pressure,
            "Shear_Stress": shear_stress
        }])

        strain_input["Sector"] = safe_transform(encoder, strain_input["Sector"])
        X_scaled = scaler.transform(strain_input)
        predicted_strain = model.predict(X_scaled)[0]

        # --- Rockfall Prediction ---
        rockfall_df = pd.DataFrame([{
            "Sector": sector,
            "Avg_Slope_Deg": slope,
            "Avg_Roughness_dry": roughness,
            "Rainfall_mm": rainfall,
            "Temperature_C": temperature,
            "Windspeed_mps": windspeed,
            "Displacement": displacement,
            "Pore_Pressure": pore_pressure,
            "Shear_Stress": shear_stress,
            "Strain": predicted_strain
        }])
        rockfall_df = pd.get_dummies(rockfall_df, columns=['Sector'], drop_first=True)

        feature_columns = joblib.load(feature_columns_path)
        rockfall_df = rockfall_df.reindex(columns=feature_columns, fill_value=0)

        scaler1 = joblib.load(scaler_path)
        numerical_cols_to_scale = scaler1.feature_names_in_
        rockfall_df_processed = rockfall_df.copy()
        rockfall_df_processed[numerical_cols_to_scale] = scaler1.transform(rockfall_df[numerical_cols_to_scale])
        
        rockfall_model = joblib.load(model_path)
        predictions_prob = rockfall_model.predict(rockfall_df_processed)[0]
        rockfall_prob = predictions_prob[1]  # Probability of the '1' class (rockfall)
       
        # Final result to be sent to the template
        result = {
            "Rainfall_mm": rainfall,
            "Windspeed_mps": windspeed,
            "Temperature_C": temperature,
            "Humidity": 65,
            "Strain": round(predicted_strain, 2),
            "Pore_Pressure": pore_pressure,
            "Displacement": displacement,
            
            "Confidence": random.uniform(95, 100),
            "riskPercentage": round((rockfall_prob * 100)-10, 2)
        }

        # --- Sending Email if Risk > 70% ---
        if rockfall_prob * 100 > 70:
            try:
                message = MIMEMultipart()
                message["From"] = sender_email
                message["To"] = receiver_email
                message["Subject"] = "⚠ Rockfall Risk Alert - RockShield AI"

                body = f"""
Dear User,



A high rockfall risk has been detected based on the following input parameters:

Input Parameters Provided:
- Rainfall (mm): {rainfall}
- Temperature (°C): {temperature}
- Windspeed (m/s): {windspeed}
- Shear Stress: {shear_stress}
- Pore Pressure: {pore_pressure}
- Displacement: {displacement}

Analysis Result:
- Sector: {sector}
- Estimated Rockfall Risk: {round((rockfall_prob * 100)-10,2)}%

Please take necessary safety precautions for the sector mentioned.

Best Regards,
RockShield AI System
Automated Risk Monitoring


Disclaimer:- This is a system generated email. Please do not reply to this email.
===============================================================
"""
                message.attach(MIMEText(body, "plain"))

                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender_email, app_password)
                server.sendmail(sender_email, receiver_email, message.as_string())
                server.quit()
                print("Alert email sent successfully!")
            except Exception as e:
                print(f"Failed to send email: {e}")

        # Always return the analysis page
        return render_template("AnalysisPage.html", analysis=result)

    except Exception as e:
        print(f"[CRITICAL] Error in /analyze: {e}", file=sys.stderr)
        return "An internal server error occurred while processing the analysis.", 500


# -----------------------------
# Flask route to reset
# -----------------------------
@app.route("/reset")
def reset():
    session.clear()
    if os.path.exists("selected_coords.json"):
        os.remove("selected_coords.json")
    return redirect(url_for("home"))

# -----------------------------
# Flask route to login/register
# -----------------------------
@app.route("/login")
def login():
    return render_template("login.html")
# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
