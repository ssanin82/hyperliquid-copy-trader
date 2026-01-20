"""
hypertrack - A Python package for tracking Hyperliquid trading activity
and detecting profitable traders using real-time WebSocket streams.
"""

__version__ = "0.1.0"

from hypertrack.recorder import TradeRecorder
from hypertrack.metrics_tracker import MetricsTracker

__all__ = ["TradeRecorder", "MetricsTracker"]
