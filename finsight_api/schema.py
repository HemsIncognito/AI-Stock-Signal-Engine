# finsight_api/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Literal

# ==============================================================================
#  REQUEST MODELS
#  Defines the shape of data coming INTO the API.
# ==============================================================================

class AnalysisRequest(BaseModel):
    """
    Defines the structure for an incoming analysis request.
    It expects a single field: the stock ticker symbol.
    """
    ticker: str = Field(
        ...,
        description="The stock ticker symbol to be analyzed (e.g., 'AAPL', 'GOOGL', 'RS').",
        examples=["RELIANCE.NS"]
    )

    country: str = Field(
        default='US',
        description="The two-letter country code for news fetching (e.g., 'US', 'IN', 'CN', 'GB').",
        examples=["IN"]
    )

# ==============================================================================
#  RESPONSE MODELS
#  Defines the shape of data going OUT of the API.
#  Using nested models makes the final response clean and well-organized.
# ==============================================================================

class SentimentOutput(BaseModel):
    """Holds the aggregated sentiment scores from news analysis."""
    dominant_sentiment: Literal["Bullish", "Bearish", "Neutral"] = Field(
        ..., 
        description="The overall dominant sentiment.",
        examples=["Bullish"]
    )
    sentiment_scores: Dict[str, float] = Field(
        ...,
        description="Dictionary of aggregated sentiment scores for each category.",
        examples=[{"Bullish": 0.81, "Bearish": 0.12, "Neutral": 0.07}]
    )

class ForecastOutput(BaseModel):
    """Contains the price trend forecast data from the time-series model."""
    prediction_days: int = Field(..., description="Number of days forecasted.", examples=[7])
    trend: Literal["Upward", "Downward", "Sideways"] = Field(
        ..., 
        description="The overall predicted trend."
    )
    forecasted_prices: List[float] = Field(
        ...,
        description="A list of forecasted closing prices for the prediction period."
    )

class InsightOutput(BaseModel):
    """Represents a single key insight extracted by the LLM from a news article."""
    source: str = Field(..., description="The source URL of the article.", examples=["https://www.reuters.com/some-article"])
    insight: str = Field(..., description="The key takeaway or summary of the article's impact.")
    impact: Literal["Positive", "Negative", "Uncertain"] = Field(
        ...,
        description="The likely impact of this insight on the stock."
    )


class AnalysisResponse(BaseModel):
    """
    Defines the final, structured JSON object returned by the /stocks/analyze endpoint.
    It consolidates all the different analysis components into a single response.
    """
    company_name: str = Field(..., description="The full name of the company.", example="Reliance Industries Limited")
    ticker: str = Field(..., description="The ticker symbol that was analyzed.", examples=["RELIANCE.NS"])
    final_recommendation: Literal["Buy", "Sell", "Hold"] = Field(
        ...,
        description="The fused, final investment signal from the intelligence layer."
    )
    recommendation_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="A confidence score (0.0 to 1.0) for the final recommendation.",
        examples=[0.85]
    )
    sentiment: SentimentOutput
    forecast: ForecastOutput
    insights: List[InsightOutput]