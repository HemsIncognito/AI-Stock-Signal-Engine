# finsight_api/services/sentiment_analyzer.py

import google.generativeai as genai
from typing import List, Dict
import json

from finsight_api.config import settings

# Use the same configured Gemini model from the intelligence service
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-2.5-flash')
    print("Sentiment Analyzer now uses Gemini model.")
except Exception as e:
    llm = None
    print(f"Error configuring Gemini model for sentiment analysis: {e}")

class SentimentAnalyzer:
    """
    A class to handle sentiment analysis of financial news using the Gemini LLM.
    """
    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """
        Analyzes a list of news articles using Gemini and returns aggregated sentiment.
        """
        if not llm or not articles:
            return self._default_sentiment("Model not loaded or no articles")

        # Combine titles and descriptions for the LLM prompt
        # We'll use the top 5 articles for a good sample size
        article_texts = [
            f"- {article['title']}: {article['description']}" for article in articles[:5]
        ]
        prompt_body = "\n".join(article_texts)

        prompt = f"""
        Analyze the sentiment of the following financial news articles from an investor's perspective.
        - Use "Bullish" for news that clearly suggests positive future performance.
        - Use "Bearish" for news that clearly suggests negative future performance.
        - Use "Neutral" for factual statements or announcements without clear positive or negative implications.
        
        Provide the output as a single JSON object with a key "sentiments" which is a list of strings. Do not add any other text or preamble.

        Example:
        {{
          "sentiments": ["Bullish", "Neutral", "Bearish"]
        }}

        News Articles:
        {prompt_body}
        """
        
        try:
            response = llm.generate_content(prompt)
            # Clean the response to ensure it's valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            sentiment_data = json.loads(cleaned_response)
            sentiments = sentiment_data.get("sentiments", [])
            
            if not sentiments:
                return self._default_sentiment("Parsed empty sentiment list from LLM")

            # Aggregate the results
            sentiment_counts = {
                "Bullish": sentiments.count("Bullish"),
                "Bearish": sentiments.count("Bearish"),
                "Neutral": sentiments.count("Neutral")
            }
            
            total_articles = len(sentiments)
            sentiment_scores = {
                label: count / total_articles for label, count in sentiment_counts.items()
            }
            
            dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            return {
                "dominant_sentiment": dominant_sentiment,
                "sentiment_scores": sentiment_scores
            }
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error during Gemini sentiment analysis: {e}")
            return self._default_sentiment("LLM analysis or parsing failed")

    def _default_sentiment(self, reason: str) -> Dict:
        """Returns a default neutral sentiment if analysis cannot be performed."""
        print(f"Warning: Returning default sentiment. Reason: {reason}")
        return {
            "dominant_sentiment": "Neutral",
            "sentiment_scores": {"Bullish": 0.0, "Bearish": 0.0, "Neutral": 1.0}
        }

# Create a single, importable instance
senti_analyzer = SentimentAnalyzer()







































# # finsight_api/services/sentiment_analyzer.py

# from transformers import pipeline
# from typing import List, Dict

# # Import the centralized settings instance
# from finsight_api.config import settings

# class SentimentAnalyzer:
#     """
#     A class to handle sentiment analysis of financial news using a FinBERT model.
    
#     The model is loaded once during initialization for efficiency.
#     """
#     def __init__(self):
#         """
#         Initializes the SentimentAnalyzer and loads the pre-trained model.
#         """
#         print("Initializing Sentiment Analyzer...")
#         try:
#             # Load the text-classification pipeline with the specified FinBERT model
#             # This will download the model from Hugging Face Hub on the first run
#             self.classifier = pipeline(
#                 "text-classification",
#                 model=settings.SENTIMENT_MODEL
#             )
#             print("Sentiment Analyzer model loaded successfully.")
#         except Exception as e:
#             print(f"Error loading sentiment model: {e}")
#             self.classifier = None

#     def analyze_articles(self, articles: List[Dict]) -> Dict:
#         """
#         Analyzes a list of news articles and returns aggregated sentiment scores.

#         Args:
#             articles: A list of article dictionaries from gnews.

#         Returns:
#             A dictionary containing the dominant sentiment and aggregated scores.
#         """
#         if not self.classifier:
#             return self._default_sentiment("Model not loaded")
        
#         if not articles:
#             return self._default_sentiment("No articles provided")

#         # Extract titles for analysis. We can combine title and description for more context.
#         texts_to_analyze = [
#             f"{article['title']}. {article['description']}" for article in articles
#         ]
        
#         try:
#             # Run the sentiment analysis on all texts at once
#             sentiments = self.classifier(texts_to_analyze)
#         except Exception as e:
#             print(f"Error during sentiment analysis: {e}")
#             return self._default_sentiment("Analysis failed")

#         # Aggregate the results
#         sentiment_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
#         for sentiment in sentiments:
#             label = sentiment['label']
#             if label in sentiment_counts:
#                 sentiment_counts[label] += 1
        
#         total_articles = len(articles)
#         sentiment_scores = {
#             label: count / total_articles for label, count in sentiment_counts.items()
#         }
        
#         # Determine the dominant sentiment
#         dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        
#         return {
#             "dominant_sentiment": dominant_sentiment,
#             "sentiment_scores": sentiment_scores
#         }
        
#     def _default_sentiment(self, reason: str) -> Dict:
#         """Returns a default neutral sentiment if analysis cannot be performed."""
#         print(f"Warning: Returning default sentiment. Reason: {reason}")
#         return {
#             "dominant_sentiment": "Neutral",
#             "sentiment_scores": {"Bullish": 0.0, "Bearish": 0.0, "Neutral": 1.0}
#         }

# # Create a single, importable instance of the analyzer
# # This pattern ensures the model is loaded only once when the module is imported
# senti_analyzer = SentimentAnalyzer()