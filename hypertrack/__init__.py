"""
hypertrack - A Python package for tracking Hyperliquid trading activity
and detecting profitable traders using real-time WebSocket streams.
"""

__version__ = "0.1.0"

from hypertrack.recorder import TradeRecorder
from hypertrack.chain_subscriber import ChainSubscriber
from hypertrack.wallet_profiler import WalletProfiler

__all__ = ["TradeRecorder", "ChainSubscriber", "WalletProfiler"]
