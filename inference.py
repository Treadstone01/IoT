import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
import warnings
import requests

# Suppress the FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

class EnergyInferencePipeline:
    """
    A pipeline for preprocessing new data and making minute-level energy predictions
    with a trained LSTM model.
    """
    def __init__(self, model_path, scaler_features_path, n_steps=60):
        print("Loading minute-level model and scalers...")
        self.model = keras.models.load_model(model_path)
        self.scaler_features = joblib.load(scaler_features_path)
        self.n_steps = n_steps
        self.feature_columns = ['Voltage', 'current', 'power', 'energy', 'PF', 'Frequency', 'var', 'pf']
        self.target_variable = 'energy'
        self.thingspeak_url = 'https://api.thingspeak.com/channels/2966741/feeds.json?api_key=8PCY8HQ6WKC7MYC0&results=8000'
        self.carbon_factor = 0.5  # kgCO₂ per kWh

    def fetch_data(self):
        """Fetches the latest data from ThingSpeak API."""
        try:
            response = requests.get(self.thingspeak_url)
            response.raise_for_status()
            data = response.json()
            feeds = data.get('feeds', [])

            if not feeds:
                raise ValueError("No feeds found in the API response.")

            df = pd.DataFrame(feeds)
            column_mapping = {
                "field1": "Voltage", "field2": "current", "field3": "power", "field4": "energy",
                "field5": "PF", "field6": "Frequency", "field7": "var", "field8": "pf"
            }
            df.rename(columns=column_mapping, inplace=True)
            df['created_at'] = pd.to_datetime(df['created_at'])

            for col in self.feature_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error fetching data from ThingSpeak API: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Error processing API response: {e}")

    def preprocess_data(self, raw_data_df):
        """Preprocesses new, raw time-series data for minute-level prediction."""
        raw_data_df.set_index('created_at', inplace=True)

        resampled_df = raw_data_df.resample('1T').agg({
            'Voltage': 'mean',
            'current': 'mean',
            'power': 'mean',
            'energy': 'max',   # cumulative counter
            'PF': 'mean',
            'Frequency': 'mean',
            'var': 'mean',
            'pf': 'mean'
        })
        resampled_df.interpolate(method='linear', inplace=True)
        resampled_df.dropna(inplace=True)

        # ⚡ Compute per-minute consumption
        resampled_df['energy_diff'] = resampled_df['energy'].diff().clip(lower=0)
        resampled_df.dropna(inplace=True)

        # Drop cumulative energy
        features_df = resampled_df.drop(columns=['energy', 'energy_diff'])

        # Scale features
        scaled_features = self.scaler_features.transform(features_df)
        scaled_features_df = pd.DataFrame(
            scaled_features, columns=features_df.columns, index=features_df.index
        )

        last_known_diff = resampled_df['energy_diff'].iloc[-1]
        last_known_agg = resampled_df['energy'].iloc[-1]

        return scaled_features_df, last_known_diff, last_known_agg

    def create_sequence(self, preprocessed_df):
        """Creates a single sequence from the last N_STEPS of the preprocessed data."""
        if len(preprocessed_df) < self.n_steps:
            raise ValueError(
                f"Not enough data to create a sequence. Need {self.n_steps} minutes, but have {len(preprocessed_df)}."
            )
        sequence = preprocessed_df.iloc[-self.n_steps:, :].values
        return np.reshape(sequence, (1, self.n_steps, preprocessed_df.shape[1]))

    def predict(self):
        """
        Performs the full pipeline and returns:
        - per-minute prediction
        - hourly & daily totals
        - carbon footprint
        - last aggregate + projected aggregate
        """
        raw_data_df = self.fetch_data()
        preprocessed_features, last_known_diff, last_known_agg = self.preprocess_data(raw_data_df)
        model_input = self.create_sequence(preprocessed_features)

        predicted_consumption = self.model.predict(model_input)[0][0]  # kWh/min

        # 🔎 Derive extended values
        hourly_total = predicted_consumption * 60
        daily_total = predicted_consumption * 1440

        carbon_min = predicted_consumption * self.carbon_factor
        carbon_hour = hourly_total * self.carbon_factor
        carbon_day = daily_total * self.carbon_factor

        projected_agg = last_known_agg + daily_total

        return {
            "prediction_kwh_per_min": float(predicted_consumption),
            "hourly_total_kwh": float(hourly_total),
            "daily_total_kwh": float(daily_total),
            "carbon_per_min_kg": float(carbon_min),
            "carbon_hourly_kg": float(carbon_hour),
            "carbon_daily_kg": float(carbon_day),
            "last_agg_energy_kwh": float(last_known_agg),
            "projected_agg_energy_kwh": float(projected_agg),
        }
