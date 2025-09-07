# finsight_api/services/intelligence_service.py

import google.generativeai as genai
from typing import List, Dict, Literal

from finsight_api.config import settings
from finsight_api.schema import SentimentOutput, ForecastOutput

# Configure the Gemini API client
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini model configured successfully.")
except Exception as e:
    llm = None
    print(f"Error configuring Gemini model: {e}")

class IntelligenceService:
    """
    This service generates qualitative insights using an LLM and fuses
    all analytical signals into a final recommendation.
    """
    def generate_insights(self, articles: List[Dict], ticker: str) -> List[Dict]:
        """
        Uses the Gemini LLM to summarize the key takeaways from news articles.
        """
        if not llm or not articles:
            print("DEBUG: No LLM configured or no articles found. Skipping insights.")
            return []

        # We'll analyze the top 3 most relevant articles to keep API calls low
        top_articles = articles[:3]
        
        # Combine titles and descriptions for the LLM prompt
        article_texts = [f"Source: {a['url']}\nTitle: {a['title']}\n" for a in top_articles]
        prompt_body = "\n---\n".join(article_texts)

        # ... inside the generate_insights method ...
        prompt = f"""
        Act as a senior financial analyst reviewing news for {ticker}. Your task is to extract specific, forward-looking information AND explain its sentiment.

        From the news articles below, identify and list the following for each relevant article:
        1.  **Key Fact:** The single most important forward-looking catalyst, metric, or strategic shift.
        2.  **Rationale:** A brief, one-sentence explanation of why this fact is positive, negative, or uncertain for an investor.

        Format the output as a numbered list. Label each part clearly. Do not add any preamble.

        Example:
        1.  Key Fact: [Key Metric] Morgan Stanley reiterated a price target of $250.
            Rationale: This indicates continued analyst confidence in the stock's upward potential. (Positive)

        News Articles:
        {prompt_body}
        """

        try:
            response = llm.generate_content(prompt)
            # (Your debug prints can stay here)

            insights = []
            
            # 1. Clean the entire response text first
            # Remove markdown bolding (**) and newlines within a single insight
            clean_text = response.text.strip().replace('**', '')
            
            # Split the text into potential insight blocks, separated by numbered lists
            insight_blocks = clean_text.split('\n')

            current_fact = ""
            current_rationale = ""

            for line in insight_blocks:
                line = line.strip()
                if not line:
                    continue

                # Check for the start of a new insight
                if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                    # If we have a complete insight from the previous lines, process it
                    if current_fact and current_rationale:
                        # (Processing logic will be here)
                        pass
                    # Reset for the new insight
                    current_fact = ""
                    current_rationale = ""

                if "Key Fact:" in line:
                    current_fact = line.replace("Key Fact:", "").strip()
                elif "Rationale:" in line:
                    current_rationale = line.replace("Rationale:", "").strip()

                # If both parts of an insight have been found, process and store it
                if current_fact and current_rationale:
                    full_insight = f"{current_fact} Rationale: {current_rationale}"
                    impact = "Uncertain"
                    if "(Positive)" in current_rationale:
                        impact = "Positive"
                    elif "(Negative)" in current_rationale:
                        impact = "Negative"

                    # Clean the impact label from the final text
                    full_insight = full_insight.replace("(Positive)", "").replace("(Negative)", "").replace("(Uncertain)", "").strip()

                    insights.append({
                        "source": "News Analysis",
                        "insight": full_insight,
                        "impact": impact
                    })
                    # Reset for the next potential insight
                    current_fact = ""
                    current_rationale = ""
            
            return insights
            
        except Exception as e:
            print(f"Error generating insights with Gemini: {e}")
            return []

    def fuse_signals(
        self,
        sentiment: Dict,
        forecast: Dict,
        insights: List[Dict]
    ) -> Dict:
        """
        Combines all analytical signals into a final recommendation.
        This uses a simple weighted scoring system.
        """
        score = 0
        confidence_factors = []

        # 1. Analyze Forecast Trend (Highest weight)
        if forecast['trend'] == "Upward":
            score += 2
        elif forecast['trend'] == "Downward":
            score -= 2
        confidence_factors.append(max(abs(p - forecast['forecasted_prices'][0]) for p in forecast['forecasted_prices']) / forecast['forecasted_prices'][0])

        # 2. Analyze Sentiment (Medium weight)
        dominant_sentiment = sentiment['dominant_sentiment']
        if dominant_sentiment == "Bullish":
            score += 1
        elif dominant_sentiment == "Bearish":
            score -= 1
        confidence_factors.append(sentiment['sentiment_scores'].get(dominant_sentiment, 0.5))
        
        # 3. Analyze LLM Insights (Fine-tuning weight)
        positive_insights = sum(1 for i in insights if i['impact'] == 'Positive')
        negative_insights = sum(1 for i in insights if i['impact'] == 'Negative')
        
        # Give each net positive insight a stronger score
        insight_score = positive_insights - negative_insights
        score += insight_score # Now, 3 positive insights would add +3 to the score

        # Determine final recommendation
        if score > 1.5:
            recommendation = "Buy"
        elif score < -1.5:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
            
        # Calculate a simple confidence score
        confidence = min(sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5, 1.0) * 100
        
        return {
            "final_recommendation": recommendation,
            "recommendation_confidence": round(confidence / 100, 2)
        }

# Create a single, importable instance
intelligence = IntelligenceService()