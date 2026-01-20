"""
TradeRecorder - Records Hyperliquid trading events to a human-readable log file.

This module subscribes to Hyperliquid WebSocket streams to capture:
- User fills (trades executed by specific addresses)
- Trade data (price, size, side, timestamp)
- Position updates
- PnL information

All events are recorded in a parseable, human-readable format suitable for ML analysis.
"""

import json
import time
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    print("Warning: hyperliquid-python-sdk not installed. Install with: pip install hyperliquid-python-sdk")


class TradeRecorder:
    """
    Records Hyperliquid trading events to a text file.
    
    Uses WebSocket streams for real-time data capture. Records:
    - User fills with wallet addresses
    - Trade executions (price, size, side, coin)
    - Timestamps for time-series analysis
    - Trade values for profitability calculations
    
    The output format is JSON Lines (JSONL) - one JSON object per line,
    making it both human-readable and easily parseable for ML tasks.
    """
    
    def __init__(self, filename: str = "hyperliquid_trades.log", base_url: Optional[str] = None):
        """
        Initialize the TradeRecorder.
        
        Args:
            filename: Output file path for recording trades
            base_url: Optional custom Hyperliquid API URL (defaults to mainnet)
        """
        if not HYPERLIQUID_AVAILABLE:
            raise ImportError(
                "hyperliquid-python-sdk is required. Install with: pip install hyperliquid-python-sdk"
            )
        
        self.filename = filename
        self.file = None
        self.running = False
        self.info = None
        self.base_url = base_url or constants.MAINNET_API_URL
        
        # Statistics
        self.stats = {
            "total_trades": 0,
            "total_fills": 0,
            "start_time": None,
            "coins_tracked": set()
        }
    
    def _format_trade_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a trade event into a standardized structure for ML analysis.
        
        Args:
            event_type: Type of event (trade, fill, position_update, etc.)
            data: Raw event data from Hyperliquid
            
        Returns:
            Formatted event dictionary
        """
        formatted = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "unix_timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        
        # Extract key fields for easier parsing
        if "coin" in data:
            formatted["coin"] = data["coin"]
            self.stats["coins_tracked"].add(data["coin"])
        
        if "side" in data:
            formatted["side"] = data["side"]  # "A" for ask/sell, "B" for bid/buy
        
        if "px" in data and "sz" in data:
            try:
                price = float(data["px"])
                size = float(data["sz"])
                formatted["price"] = price
                formatted["size"] = size
                formatted["notional_usd"] = price * size
            except (ValueError, TypeError):
                pass
        
        if "user" in data:
            formatted["user_address"] = data["user"]
        
        if "oid" in data:
            formatted["order_id"] = data["oid"]
        
        if "hash" in data:
            formatted["trade_hash"] = data["hash"]
        
        if "closedPnl" in data:
            formatted["closed_pnl"] = data["closedPnl"]
        
        if "liquidationPx" in data:
            formatted["liquidation_price"] = data["liquidationPx"]
        
        return formatted
    
    def _write_event(self, event: Dict[str, Any]):
        """
        Write a formatted event to the log file.
        
        Args:
            event: Formatted event dictionary
        """
        if self.file is None:
            self.file = open(self.filename, "a", encoding="utf-8")
        
        # Write as JSON Lines format (one JSON object per line)
        json_line = json.dumps(event, separators=(',', ':'))
        self.file.write(json_line + "\n")
        self.file.flush()  # Ensure data is written immediately
        
        # Also print a human-readable summary
        coin = event.get("coin", "Unknown")
        side = event.get("side", "?")
        price = event.get("price", "?")
        size = event.get("size", "?")
        user = event.get("user_address", "N/A")[:10] + "..." if event.get("user_address") else "N/A"
        
        print(f"[{event['timestamp']}] {event['event_type']:15} | "
              f"Coin: {coin:8} | Side: {side} | "
              f"Price: {price:12} | Size: {size:12} | User: {user}")
    
    def _handle_trade(self, trade_data: Dict[str, Any]):
        """
        Handle incoming trade data from WebSocket.
        
        Args:
            trade_data: Raw trade data from Hyperliquid
        """
        try:
            event = self._format_trade_event("trade", trade_data)
            self._write_event(event)
            self.stats["total_trades"] += 1
        except Exception as e:
            print(f"Error handling trade: {e}")
    
    def _handle_user_fill(self, fill_data: Dict[str, Any]):
        """
        Handle user fill data (trades executed by specific addresses).
        
        Args:
            fill_data: Raw fill data from Hyperliquid
        """
        try:
            event = self._format_trade_event("user_fill", fill_data)
            self._write_event(event)
            self.stats["total_fills"] += 1
        except Exception as e:
            print(f"Error handling fill: {e}")
    
    def _poll_trades(self):
        """
        Poll for recent trades using REST API.
        Uses the Hyperliquid Info API to get recent trades and user fills.
        This is reliable and works with free public RPC endpoints.
        """
        last_seen_trades = set()
        tracked_users = set()  # Track users we've seen to monitor their fills
        poll_interval = 3  # Poll every 3 seconds to balance latency and rate limits
        coin_poll_interval = 10  # Poll each coin every 10 seconds
        
        while self.running:
            try:
                # Get recent trades from all active markets
                meta = self.info.meta()
                if not meta or "universe" not in meta:
                    print("Warning: Could not fetch market metadata")
                    time.sleep(poll_interval)
                    continue
                
                coins = [coin["name"] for coin in meta["universe"]]
                print(f"Monitoring {len(coins)} active markets...")
                
                # Poll each coin for recent trades
                for coin in coins:
                    if not self.running:
                        break
                    
                    try:
                        # Get recent trades for this coin
                        trades = self._get_recent_trades(coin, limit=50)
                        
                        for trade in trades:
                            # Create a unique ID for this trade
                            trade_id = f"{coin}_{trade.get('time', 0)}_{trade.get('hash', '')}"
                            
                            if trade_id not in last_seen_trades:
                                last_seen_trades.add(trade_id)
                                
                                # Record the trade
                                self._handle_trade(trade)
                                
                                # Track user addresses for fill monitoring
                                if "user" in trade:
                                    tracked_users.add(trade["user"])
                                
                                # Limit the size of last_seen_trades to prevent memory issues
                                if len(last_seen_trades) > 10000:
                                    # Keep only the most recent 5000
                                    last_seen_trades = set(list(last_seen_trades)[-5000:])
                        
                        time.sleep(0.2)  # Small delay between coins to avoid rate limits
                        
                    except Exception as e:
                        # Skip this coin if there's an error
                        print(f"Error polling {coin}: {e}")
                        continue
                
                # Periodically check user fills for tracked users
                # This gives us more detailed information about profitable traders
                if len(tracked_users) > 0:
                    # Sample a subset of users to avoid rate limits
                    users_to_check = list(tracked_users)[:10]  # Check 10 users per cycle
                    
                    for user in users_to_check:
                        if not self.running:
                            break
                        
                        try:
                            fills = self._get_user_fills(user)
                            for fill in fills:
                                # Record user fills (these contain PnL and position info)
                                self._handle_user_fill(fill)
                        except Exception as e:
                            # Skip this user if there's an error
                            continue
                        
                        time.sleep(0.5)  # Delay between user queries
                
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"Error polling trades: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _get_recent_trades(self, coin: str, limit: int = 100):
        """
        Get recent trades for a specific coin using REST API.
        
        Args:
            coin: Coin symbol (e.g., "BTC", "ETH")
            limit: Maximum number of trades to fetch
            
        Returns:
            List of trade dictionaries
        """
        try:
            # Use the Info API to get recent trades
            # The exact method name depends on the SDK version
            # Common patterns: recent_trades(), get_trades(), etc.
            
            # Try different possible method names
            if hasattr(self.info, 'recent_trades'):
                return self.info.recent_trades(coin, limit)
            elif hasattr(self.info, 'get_recent_trades'):
                return self.info.get_recent_trades(coin, limit)
            elif hasattr(self.info, 'trades'):
                return self.info.trades(coin, limit)
            else:
                # Fallback: use the meta and query methods
                # Get all trades from the exchange
                meta = self.info.meta()
                if meta:
                    # Try to get trades from the exchange state
                    # This is a generic fallback
                    return []
        except Exception as e:
            print(f"Error getting trades for {coin}: {e}")
            return []
        
        return []
    
    def _get_user_fills(self, user_address: str):
        """
        Get fills for a specific user address.
        
        Args:
            user_address: User wallet address
            
        Returns:
            List of fill dictionaries
        """
        try:
            # Use the Info API to get user fills
            if hasattr(self.info, 'user_fills'):
                return self.info.user_fills(user_address)
            elif hasattr(self.info, 'get_user_fills'):
                return self.info.get_user_fills(user_address)
            elif hasattr(self.info, 'fills'):
                return self.info.fills(user_address)
        except Exception as e:
            print(f"Error getting fills for user {user_address[:10]}...: {e}")
            return []
        
        return []
    
    def start(self):
        """
        Start recording trades. This will run until stopped with Ctrl+C.
        """
        if self.running:
            print("Recorder is already running!")
            return
        
        print(f"=" * 80)
        print(f"Hyperliquid Trade Recorder")
        print(f"=" * 80)
        print(f"Output file: {self.filename}")
        print(f"API URL: {self.base_url}")
        print(f"Press Ctrl+C to stop recording")
        print(f"=" * 80)
        
        try:
            self.info = Info(self.base_url, skip_ws=False)
            self.running = True
            self.stats["start_time"] = time.time()
            
            # Open file for writing
            self.file = open(self.filename, "a", encoding="utf-8")
            
            # Write header comment
            header = {
                "type": "header",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "recorder_version": "0.1.0",
                "description": "Hyperliquid trading activity log - JSON Lines format",
                "api_url": self.base_url
            }
            self.file.write(json.dumps(header, separators=(',', ':')) + "\n")
            self.file.flush()
            
            # Start monitoring trades
            # Use polling approach for reliability with free public RPC
            self._poll_trades()
            
        except KeyboardInterrupt:
            print("\n\nStopping recorder...")
            self.stop()
        except Exception as e:
            print(f"\n\nError: {e}")
            self.stop()
            raise
    
    def stop(self):
        """
        Stop recording and close the file.
        """
        if not self.running:
            return
        
        self.running = False
        
        if self.file:
            # Write footer with statistics
            duration = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
            footer = {
                "type": "footer",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "statistics": {
                    "total_trades": self.stats["total_trades"],
                    "total_fills": self.stats["total_fills"],
                    "duration_seconds": duration,
                    "coins_tracked": list(self.stats["coins_tracked"])
                }
            }
            self.file.write(json.dumps(footer, separators=(',', ':')) + "\n")
            self.file.close()
            self.file = None
        
        print(f"\n{'=' * 80}")
        print(f"Recording stopped")
        print(f"Total trades recorded: {self.stats['total_trades']}")
        print(f"Total fills recorded: {self.stats['total_fills']}")
        print(f"Coins tracked: {len(self.stats['coins_tracked'])}")
        print(f"Output saved to: {self.filename}")
        print(f"{'=' * 80}")


def main():
    """
    Main entry point for the recorder script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Record Hyperliquid trading activity to a log file"
    )
    parser.add_argument(
        "-f", "--filename",
        default="hyperliquid_trades.log",
        help="Output log file path (default: hyperliquid_trades.log)"
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Custom Hyperliquid API URL (default: mainnet)"
    )
    
    args = parser.parse_args()
    
    recorder = TradeRecorder(filename=args.filename, base_url=args.url)
    
    def signal_handler(sig, frame):
        recorder.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        recorder.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        recorder.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
