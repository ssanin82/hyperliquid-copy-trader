"""
hypertrack - A Python package for tracking Hyperliquid trading activity
and detecting profitable traders using real-time WebSocket streams.
"""

__version__ = "0.1.0"

from hypertrack.metrics_tracker import MetricsTracker
from hypertrack.recorder import TradeRecorder
from hypertrack.chain_test import ChainSubscriber
from hypertrack.constants import TOTAL_TRADES_THRESHOLD, WATCH_COINS, TRADE_SAMPLE_SIZE

__all__ = ["MetricsTracker", "TradeRecorder", "ChainSubscriber", "TOTAL_TRADES_THRESHOLD", "WATCH_COINS", "TRADE_SAMPLE_SIZE"]
