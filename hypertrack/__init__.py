"""
hypertrack - A Python package for tracking Hyperliquid trading activity
and detecting profitable traders using real-time WebSocket streams.
"""

__version__ = "0.1.0"

from hypertrack.metrics_tracker import MetricsTracker
from hypertrack.recorder import TradeRecorder
from hypertrack.chain_subscriber import ChainSubscriber
from hypertrack.wallet_profiler import WalletProfiler
from hypertrack.constants import TOTAL_TRADES_THRESHOLD, WATCH_COINS, TRADE_SAMPLE_SIZE

__all__ = ["MetricsTracker", "TradeRecorder", "ChainSubscriber", "WalletProfiler", "TOTAL_TRADES_THRESHOLD", "WATCH_COINS", "TRADE_SAMPLE_SIZE"]
