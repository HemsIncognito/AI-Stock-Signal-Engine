# training/preprocessor.py

import pandas as pd
import yfinance as yf
from typing import Tuple

def fetch_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetches historical stock data from yfinance."""
    print(f"Fetching {period} of historical data for {ticker}...")
    stock_data = yf.Ticker(ticker).history(period=period)
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}. It may be delisted or invalid.")
    return stock_data

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the dataframe with common technical indicators.
    
    Args:
        df: A pandas DataFrame with stock prices (must include Open, High, Low, Close, Volume).
        
    Returns:
        The DataFrame with added indicator columns.
    """
    print("Adding technical indicators...")
    
    # Simple Moving Averages (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['SMA_20']
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Drop rows with NaN values created by rolling windows
    df.dropna(inplace=True)
    
    return df

def create_sequences(
    data: pd.DataFrame, 
    sequence_length: int = 60, 
    future_days: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates sequences of historical data to predict a future value.
    
    Args:
        data: DataFrame of stock prices and indicators.
        sequence_length: Number of past days in each input sequence.
        future_days: How many days in the future to predict.
        
    Returns:
        A tuple of (X, y) where X are the input sequences and y are the target values.
    """
    print("Creating training sequences...")
    X, y = [], []
    # We will predict the 'Close' price
    target_col_index = data.columns.get_loc('Close')
    
    for i in range(len(data) - sequence_length - future_days + 1):
        # Input sequence (e.g., days 0-59)
        X.append(data.iloc[i:(i + sequence_length)].values)
        # Target value (e.g., day 60's closing price)
        y.append(data.iloc[i + sequence_length + future_days - 1, target_col_index])
        
    return pd.DataFrame(X), pd.Series(y)

# Example of how to use this preprocessor
if __name__ == '__main__':
    ticker_symbol = "AAPL"
    raw_data = fetch_data(ticker_symbol)
    processed_data = add_technical_indicators(raw_data.copy())
    
    print("\n--- Raw Data Head ---")
    print(raw_data.head())
    
    print("\n--- Processed Data with Indicators Head ---")
    print(processed_data.head())
    
    # You would then pass 'processed_data' to the create_sequences function
    # before scaling and training.
    
    print(f"\nData processing complete for {ticker_symbol}.")