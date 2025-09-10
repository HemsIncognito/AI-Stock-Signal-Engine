# finsight_api/config.py

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings and secrets using Pydantic.
    It automatically reads environment variables from a .env file.
    """
    
    # Load environment variables from a .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Gemini API Key ---
    # The API key for Google's Gemini service.
    # It's critical for the insight generation module.
    GEMINI_API_KEY: str = "API_KEY_NOT_SET"

    # --- FinSight API Key ---
    # The API key required to access the FinSight API endpoints.
    FINSIGHT_API_KEY: str = "API_KEY_NOT_SET"
    
    # --- Data Fetching Defaults ---
    # Default time periods for fetching historical data and news.
    DEFAULT_PRICE_HISTORY_PERIOD: str = "2y"
    DEFAULT_NEWS_DURATION: str = "14d"

    # --- Forecasting Parameters ---
    # Defines how many days into the future the model will predict.
    FORECAST_DAYS: int = 10


# Create a single, importable instance of the settings
settings = Settings()