# finsight_api/services/trends_forecaster.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

from finsight_api.config import settings

import sys
sys.path.append('../')
from training import preprocessor

class TrendForecaster:
    """
    A class to forecast stock price trends by training a multivariate LSTM model in real-time.
    """
    def __init__(self):
        self.sequence_length = 60  # Number of past days to consider for each prediction

    def forecast_trend(self, price_history: pd.DataFrame) -> Dict:
        """
        Orchestrates the real-time training and forecasting process.
        """
        # 1. Add technical indicators
        data = preprocessor.add_technical_indicators(price_history.copy())
        
        # 2. Scale data and create sequences for training
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X_train, y_train = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X_train.append(scaled_data[i-self.sequence_length:i])
            y_train.append(scaled_data[i, data.columns.get_loc('Close')])
        X_train, y_train = np.array(X_train), np.array(y_train)

        if len(X_train) == 0:
            raise ValueError("Not enough historical data to train the forecasting model.")

        # 3. Build and train the LSTM model on-the-fly
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=50, return_sequences=True),
            Dropout(0.3),
            LSTM(units=50, return_sequences=False),
            Dropout(0.3),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(f"Training LSTM model in real-time for {price_history.attrs.get('info', {}).get('symbol', '')}...")
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
        
        # 4. Predict future prices iteratively
        last_sequence = scaled_data[-self.sequence_length:]
        future_predictions_scaled = []
        current_sequence = last_sequence.reshape((1, self.sequence_length, scaled_data.shape[1]))

        for _ in range(settings.FORECAST_DAYS):
            next_pred_scaled = model.predict(current_sequence, verbose=0)[0]
            future_predictions_scaled.append(next_pred_scaled[0])
            
            new_row_placeholder = current_sequence[0, -1, :].copy()
            new_row_placeholder[data.columns.get_loc('Close')] = next_pred_scaled[0]
            new_input = np.append(current_sequence[0, 1:, :], [new_row_placeholder], axis=0)
            current_sequence = new_input.reshape((1, self.sequence_length, scaled_data.shape[1]))

        # 5. Inverse transform and determine trend
        dummy_array = np.zeros((len(future_predictions_scaled), scaled_data.shape[1]))
        dummy_array[:, data.columns.get_loc('Close')] = np.array(future_predictions_scaled).flatten()
        predicted_prices = scaler.inverse_transform(dummy_array)[:, data.columns.get_loc('Close')].tolist()
        
        last_actual_price = price_history['Close'].iloc[-1]
        last_predicted_price = predicted_prices[-1]
        
        if last_predicted_price > last_actual_price * 1.01:
            trend = "Upward"
        elif last_predicted_price < last_actual_price * 0.99:
            trend = "Downward"
        else:
            trend = "Sideways"

        return {
            "prediction_days": settings.FORECAST_DAYS,
            "trend": trend,
            "forecasted_prices": [round(price, 2) for price in predicted_prices]
        }

# Create a single, importable instance
trend_forecaster = TrendForecaster()