# evaluation/config.py

# A diverse portfolio of stocks from different sectors to test our models against.
STOCKS_TO_TEST = [
    "AAPL",  # Technology
    "MSFT",  # Technology
    "JPM",   # Finance
    "JNJ",   # Healthcare
    "F",     # Automotive
    "NVDA"   # Semiconductors
]

# Backtesting parameters
TEST_YEARS = 1
SEQUENCE_LENGTH = 60
PERIOD = "5y"