import asyncio
import os
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings

from energy_inference_pipeline import EnergyInferencePipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# Define response schema
class PredictionResponse(BaseModel):
    status: str
    prediction_kwh_per_min: float
    hourly_total_kwh: float
    daily_total_kwh: float
    carbon_per_min_kg: float
    carbon_hourly_kg: float
    carbon_daily_kg: float
    last_agg_energy_kwh: float
    projected_agg_energy_kwh: float

# Initialize FastAPI
app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load inference pipeline
try:
    pipeline = EnergyInferencePipeline(
        model_path=os.path.join(os.getcwd(), "minute_lstm_model.h5"),
        scaler_features_path=os.path.join(os.getcwd(), "scaler_features.joblib"),
        n_steps=60
    )
    print("EnergyInferencePipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load pipeline: {e}")
    pipeline = None

# Store the latest prediction
latest_prediction = None

async def prediction_loop():
    """Background loop to update predictions every minute."""
    global latest_prediction
    while True:
        if pipeline:
            try:
                pred_dict = pipeline.predict()
                latest_prediction = PredictionResponse(status="success", **pred_dict)
                print(f"Updated Prediction: {latest_prediction.dict()}")
            except Exception as e:
                print(f"Prediction failed: {e}")
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Start background prediction loop on startup."""
    asyncio.create_task(prediction_loop())

@app.get("/latest", response_model=PredictionResponse)
async def get_latest_prediction():
    """Return the most recent prediction."""
    if latest_prediction is None:
        raise HTTPException(status_code=503, detail="Prediction not available yet.")
    return latest_prediction

@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """Send continuous predictions to WebSocket clients."""
    await websocket.accept()
    while True:
        if latest_prediction:
            await websocket.send_json(latest_prediction.dict())
        await asyncio.sleep(60)
