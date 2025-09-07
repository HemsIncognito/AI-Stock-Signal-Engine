# finsight_api/services/data_fetcher.py

import yfinance as yf
from gnews import GNews
import pandas as pd
from typing import List, Dict

# Import the centralized settings instance
from finsight_api.config import settings

def get_stock_prices(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetches historical stock price data for a given ticker.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., 'AAPL').

    Returns:
        A pandas DataFrame with historical price data.
        
    Raises:
        ValueError: If the ticker is invalid or no data is found.
    """
    print(f"Fetching price history for {ticker_symbol} for the period: {settings.DEFAULT_PRICE_HISTORY_PERIOD}")
    
    ticker = yf.Ticker(ticker_symbol)
    
    # Fetch historical data using the period defined in config
    hist = ticker.history(period=settings.DEFAULT_PRICE_HISTORY_PERIOD)
    
    if hist.empty:
        raise ValueError(f"Invalid ticker or no data found for '{ticker_symbol}'. Please check the symbol.")
        
    return hist


def get_latest_news(ticker_symbol: str, country_code: str) -> List[Dict]:
    """
    Fetches and merges recent news articles from both GNews and yfinance.
    """
    print(f"Fetching news for '{ticker_symbol}' from GNews (country: {country_code}) and yfinance...")
    
    # --- 1. Fetch from GNews ---
    gnews_articles = []
    try:
        google_news = GNews(
            language='en', 
            country=country_code, 
            period=settings.DEFAULT_NEWS_DURATION
        )
        gnews_articles = google_news.get_news(ticker_symbol)
    except Exception as e:
        print(f"Warning: Could not fetch news from GNews. Error: {e}")

    # --- 2. Fetch from yfinance ---
    yfinance_articles = []
    try:
        ticker = yf.Ticker(ticker_symbol)
        # The .news attribute returns a list of article dictionaries
        yfinance_articles = ticker.news
    except Exception as e:
        print(f"Warning: Could not fetch news from yfinance. Error: {e}")

    # --- 3. Merge and De-duplicate ---
    combined_articles = []
    seen_titles = set()

    # Standardize and add articles from yfinance
    for article in yfinance_articles:
        title = article.get('title')
        if title and title not in seen_titles:
            combined_articles.append({
                'title': title,
                'description': article.get('summary', 'No description available.'),
                'url': article.get('link', '#')
            })
            seen_titles.add(title)

    # Standardize and add articles from GNews
    for article in gnews_articles:
        title = article.get('title')
        if title and title not in seen_titles:
            combined_articles.append({
                'title': title,
                'description': article.get('description', 'No description available.'),
                'url': article.get('url', '#')
            })
            seen_titles.add(title)
            
    if not combined_articles:
        print(f"Warning: No news articles found for ticker '{ticker_symbol}'.")

    return combined_articles