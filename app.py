import joblib
import pandas as pd
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import warnings

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Energy Prediction API",
    description="API for predicting energy consumption using an ARIMA model."
)

# Load the trained model
try:
    # Use joblib to load the model
    model = joblib.load('./models/arima_model.joblib')
except FileNotFoundError:
    print("Error: The model file was not found. Please run the training script first.")
    model = None
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    model = None

class PredictionResponse(BaseModel):
    status: str
    predicted_energy_kWh: float

def fetch_data():
    """
    Fetches the latest data from the ThingSpeak API.
    
    Returns:
        pd.DataFrame: A DataFrame containing the fetched data.
    """
    try:
        thingspeak_url = 'https://api.thingspeak.com/channels/2966741/feeds.json?api_key=8PCY8HQ6WKC7MYC0&results=8000'
        response = requests.get(thingspeak_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        feeds = data.get('feeds', [])
        
        if not feeds:
            raise ValueError("No feeds found in the API response.")
            
        df = pd.DataFrame(feeds)

        # Rename columns to match the model's training data
        column_mapping = {
            'field1': 'voltage',
            'field2': 'current',
            'field3': 'power',
            'field4': 'energy',
            'field5': 'frequency',
            'field6': 'power_factor',
            'field7': 'apparent_power',
            'field8': 'reactive_power'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert 'created_at' to datetime and features to numeric
        df['created_at'] = pd.to_datetime(df['created_at'])
        for col in column_mapping.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ThingSpeak API: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing API response: {e}")
        return None

def preprocess_data(raw_data_df):
    """
    Preprocesses new, raw time-series data for hourly prediction.
    """
    if raw_data_df is None:
        return None
    
    raw_data_df.set_index('created_at', inplace=True)
    
    # Handle the case of the last entry having nulls by filling with previous values
    raw_data_df.iloc[-1] = raw_data_df.iloc[-1].fillna(method='ffill')
    
    # Resample to hourly frequency and aggregate the data
    hourly_data = raw_data_df.resample('1h').agg({
        'energy': 'sum',
    })
    
    # Interpolate to fill any missing hourly values
    hourly_data.interpolate(method='linear', inplace=True)
    hourly_data.dropna(inplace=True)
    
    return hourly_data

@app.post("/predict", response_model=PredictionResponse)
def predict():
    """
    Fetches the latest data, preprocesses it, and predicts the next hour's energy consumption.
    """
    if model is None:
        return PredictionResponse(status="error", predicted_energy_kWh=0.0)

    try:
        raw_data_df = fetch_data()
        if raw_data_df is None:
            raise ValueError("Could not fetch data from the API.")
        
        preprocessed_data = preprocess_data(raw_data_df)
        if preprocessed_data is None:
            raise ValueError("Could not preprocess data.")
        
        # Use the loaded model to make a prediction. The model handles the input data internally.
        prediction = model.predict(n_periods=1)[0]
        
        # Clamp the prediction to be non-negative
        final_prediction = max(0, prediction)

        return PredictionResponse(
            status="success",
            predicted_energy_kWh=final_prediction
        )
    except Exception as e:
        return {"status": "error", "predicted_energy_kWh": None, "detail": str(e)}
