"""
Constants for hypertrack package.
"""

# Minimum number of trades required for a user to be included in the metrics report
TOTAL_TRADES_THRESHOLD = 15

# Coins to monitor for trading activity
WATCH_COINS = ["BTC", "ETH", "SOL", "ARB", "AVAX"]

# Bot score threshold - users with bot_score above this will be labeled as "Bot"
BOT_SCORE = 0.8

# Number of trades to record before stopping
TRADE_SAMPLE_SIZE = 10000

# Token to record trades for
TOKEN = "ETH"
