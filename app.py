import os
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Define a Pydantic model for the response
class PredictionResponse(BaseModel):
    status: str
    predicted_energy_kWh: float
    co2_consumption_kg: float = 0.0

# Initialize FastAPI app
app = FastAPI()

def fetch_data():
    """Fetches energy consumption data from ThingSpeak."""
    try:
        # URL to fetch the last 8000 entries with the hardcoded API key
        url = 'https://api.thingspeak.com/channels/2966741/feeds.json?api_key=8PCY8HQ6WKC7MYC0&results=8000'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('feeds', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing API response: {e}")
        return None

def preprocess_data(feeds):
    """
    Prepares data for the model by handling 'nan' values and resampling.
    """
    if not feeds:
        return None

    df = pd.DataFrame(feeds)
    df = df[['created_at', 'field4']]
    df.columns = ['timestamp', 'energy_consumption']

    # Convert to datetime and numeric, coercing errors
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['energy_consumption'] = pd.to_numeric(df['energy_consumption'], errors='coerce')
    df = df.set_index('timestamp')

    # Resample to hourly frequency and aggregate by summing the energy consumption
    hourly_data = df.resample('h').sum()
    hourly_data.interpolate(method='linear', inplace=True)
    hourly_data.dropna(inplace=True)

    return hourly_data

# Load the trained model and scalers
model = None
try:
    # Look for the model file in the same directory
    model_path = os.path.join(os.getcwd(), 'arima_model.joblib')
    model = joblib.load(model_path)
    print("ARIMA model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # The API will be in a non-functional state without the model,
    # but we can still show a friendly message.

@app.get("/predict", response_model=PredictionResponse)
def predict_energy():
    """
    Predicts the next hour's energy and CO2 consumption.
    """
    if model is None:
        raise HTTPException(status_code=500, detail={"status": "error", "predicted_energy_kWh": 0.0, "co2_consumption_kg": 0.0, "message": "Model not loaded. Please train the model and ensure the file is in the correct directory."})

    feeds = fetch_data()
    if not feeds:
        raise HTTPException(status_code=500, detail={"status": "error", "predicted_energy_kWh": 0.0, "co2_consumption_kg": 0.0, "message": "Could not fetch data from ThingSpeak."})
    
    try:
        df = preprocess_data(feeds)
        if df is None or len(df) < 24:
            raise HTTPException(status_code=500, detail={"status": "error", "predicted_energy_kWh": 0.0, "co2_consumption_kg": 0.0, "message": "Not enough data for prediction. Need at least 24 hours of data."})
            
        # Make a single-step forecast
        forecast = model.predict(n_periods=1)

        # Ensure the prediction is not negative
        predicted_value = max(0, forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0])

        # Calculate CO2 consumption (0.5 kg per kWh)
        co2_consumption = predicted_value * 0.5

        return {"status": "success", "predicted_energy_kWh": round(predicted_value, 2), "co2_consumption_kg": round(co2_consumption, 2)}
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "predicted_energy_kWh": 0.0, "co2_consumption_kg": 0.0, "message": f"An error occurred during prediction: {e}"})
