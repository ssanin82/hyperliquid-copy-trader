"""
hypertrack - A Python package for tracking Hyperliquid trading activity
and detecting profitable traders using real-time WebSocket streams.
"""

__version__ = "0.1.0"

from hypertrack.metrics_tracker import MetricsTracker
from hypertrack.constants import TOTAL_TRADES_THRESHOLD

__all__ = ["MetricsTracker", "TOTAL_TRADES_THRESHOLD"]
