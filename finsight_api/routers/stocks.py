# finsight_api/routers/stocks.py

from fastapi import APIRouter, HTTPException, status, Security, Depends
from fastapi.security import APIKeyHeader

# Import API Key security scheme
from finsight_api.config import settings

# Import our Pydantic schemas
from finsight_api.schema import AnalysisRequest, AnalysisResponse

# Import all our service singletons
from finsight_api.services import data_fetcher
from finsight_api.services.sentiment_analyzer import senti_analyzer
from finsight_api.services.trends_forecaster import trend_forecaster
from finsight_api.services.intelligence_service import intelligence

# --- API Key Security ---
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Checks if the provided API key is valid."""
    if api_key == settings.FINSIGHT_API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )
# --- End Security ---

# Create an API router
router = APIRouter(
    prefix="/stocks",
    tags=["Stock Analysis"]
)

# Define the stock analysis endpoint (with security dependency)
@router.post("/analyze", response_model=AnalysisResponse,dependencies=[Depends(get_api_key)])
async def analyze_stock(request: AnalysisRequest):
    """
    Receives a stock ticker, orchestrates the multi-pronged analysis,
    and returns a structured JSON response with actionable signals.
    """
    ticker = request.ticker.upper()
    country = request.country.upper()

    try:
        # 1. Fetch all necessary data
        print(f"[{ticker}] ==> Starting analysis...")
        price_history = data_fetcher.get_stock_prices(ticker)
        # Use company name for a better news search
        company_name = price_history.attrs.get('info', {}).get('longName', ticker)
        # # Pass the company & country to news fetcher
        news_articles = data_fetcher.get_latest_news(ticker, country)
        
        # 2. Run parallelizable analyses
        print(f"[{ticker}] ==> Analyzing sentiment and forecasting trend...")
        sentiment_result = senti_analyzer.analyze_articles(news_articles)
        forecast_result = trend_forecaster.forecast_trend(price_history)
        
        # 3. Generate qualitative insights using the LLM
        print(f"[{ticker}] ==> Generating LLM insights...")
        insights_result = intelligence.generate_insights(news_articles, ticker)

        # 4. Fuse all signals into a final recommendation
        print(f"[{ticker}] ==> Fusing signals for final recommendation...")
        fusion_result = intelligence.fuse_signals(
            sentiment=sentiment_result,
            forecast=forecast_result,
            insights=insights_result
        )

        # 5. Assemble and return the final structured response
        print(f"[{ticker}] ==> Analysis complete.")
        return AnalysisResponse(
            company_name=company_name,
            ticker=ticker,
            final_recommendation=fusion_result["final_recommendation"],
            recommendation_confidence=fusion_result["recommendation_confidence"],
            sentiment=sentiment_result,
            forecast=forecast_result,
            insights=insights_result
        )

    except ValueError as e:
        # This catches invalid tickers from the data_fetcher
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # Generic catch-all for any other unexpected errors
        print(f"An unexpected error occurred for ticker {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during analysis: {e}"
        )