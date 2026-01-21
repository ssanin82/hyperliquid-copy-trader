"""
TradeRecorder - Records trades for a specific token until sample size is reached.

Records all trade information for the specified token (defaults to TOKEN)
and stops when TRADE_SAMPLE_SIZE is reached.
"""

# Constants
TOKEN = "ETH"
TRADE_SAMPLE_SIZE = 100000

import json
import time
import signal
import sys
from datetime import datetime, UTC
from typing import Optional, Dict, Any
import websocket
import threading


class TradeRecorder:
    """Records trades for a specific symbol until sample size is reached."""
    
    MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
    
    def __init__(self, token: str = None, output_file: str = "trades.log", testnet: bool = False):
        """
        Initialize the trade recorder.
        
        Args:
            token: Token to record trades for (defaults to TOKEN)
            output_file: File to write trades to
            testnet: Use testnet endpoints
        """
        self.token = token if token is not None else TOKEN
        self.output_file = output_file
        self.testnet = testnet
        self.ws_url = self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL
        
        self.ws = None
        self.ws_thread = None
        self.running = False
        self.trade_count = 0
        self.file_handle = None
        
    def _record_trade(self, trade: Dict[str, Any]):
        """Record a trade to the output file."""
        try:
            # Format trade with timestamp
            trade_record = {
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "unix_timestamp": time.time(),
                "trade_number": self.trade_count + 1,
                "data": trade
            }
            
            # Write as JSON line
            self.file_handle.write(json.dumps(trade_record, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
            self.trade_count += 1
            
            # Print progress every 100 trades
            if self.trade_count % 10 == 0:
                print(f"Recorded {self.trade_count}/{TRADE_SAMPLE_SIZE} trades...")
            
            # Check if we've reached the sample size
            if self.trade_count >= TRADE_SAMPLE_SIZE:
                print(f"\nReached sample size of {TRADE_SAMPLE_SIZE} trades. Stopping...")
                self.stop()
                
        except Exception as e:
            print(f"Error recording trade: {e}")
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle trade messages
            if "channel" in data and data["channel"] == "trades":
                if "data" in data:
                    for trade in data["data"]:
                        # Check if this trade is for our token
                        trade_coin = trade.get("coin", "")
                        if trade_coin == self.token:
                            if not self.running:
                                return
                            self._record_trade(trade)
            
            # Also handle direct trade data
            elif "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if "coin" in item and "side" in item:
                        trade_coin = item.get("coin", "")
                        if trade_coin == self.token:
                            if not self.running:
                                return
                            self._record_trade(item)
                            
        except json.JSONDecodeError as e:
            print(f"Error parsing WebSocket message: {e}")
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
    
    def _on_ws_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.running and self.trade_count < TRADE_SAMPLE_SIZE:
            time.sleep(2)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        print(f"WebSocket connected to Hyperliquid")
        print(f"Subscribing to trades for {self.token}...")
        
        try:
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": self.token}
            }
            ws.send(json.dumps(subscribe_msg))
            print(f"Subscribed to trades for {self.token}. Waiting for trades...")
        except Exception as e:
            print(f"Error subscribing to trades: {e}")
    
    def _start_websocket(self):
        """Start WebSocket connection."""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close,
                on_open=self._on_ws_open
            )
            
            def run_ws():
                self.ws.run_forever()
            
            self.ws_thread = threading.Thread(target=run_ws, daemon=True)
            self.ws_thread.start()
        except Exception as e:
            print(f"Error starting WebSocket: {e}")
    
    def start(self):
        """Start recording trades."""
        # Open output file
        try:
            self.file_handle = open(self.output_file, 'w')
            
            # Write header
            header = {
                "type": "header",
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "token": self.token,
                "sample_size": TRADE_SAMPLE_SIZE,
                "description": f"Hyperliquid trade log for {self.token} - JSON Lines format"
            }
            self.file_handle.write(json.dumps(header, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
        except Exception as e:
            print(f"Error opening output file: {e}")
            return
        
        print("=" * 80)
        print("TRADE RECORDER")
        print("=" * 80)
        print(f"Token: {self.token}")
        print(f"Output file: {self.output_file}")
        print(f"Sample size: {TRADE_SAMPLE_SIZE} trades")
        print(f"Recording all trades for {self.token}...")
        print("Press Ctrl+C to stop early")
        print("=" * 80)
        
        self.running = True
        
        # Start WebSocket
        self._start_websocket()
        
        try:
            # Keep running until stopped or sample size reached
            while self.running and self.trade_count < TRADE_SAMPLE_SIZE:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop recording and close connections."""
        print("\nStopping recorder...")
        self.running = False
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
        
        # Close file
        if self.file_handle:
            # Write footer
            footer = {
                "type": "footer",
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "total_trades_recorded": self.trade_count,
                "sample_size": TRADE_SAMPLE_SIZE,
                "completed": self.trade_count >= TRADE_SAMPLE_SIZE
            }
            self.file_handle.write(json.dumps(footer, separators=(',', ':')) + "\n")
            self.file_handle.close()
            self.file_handle = None
        
        print(f"Recorded {self.trade_count} trades")
        print(f"Output saved to: {self.output_file}")


def main():
    """Main entry point."""
    # Set up signal handler
    recorder = TradeRecorder()
    
    def signal_handler(sig, frame):
        recorder.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start recording
    recorder.start()


if __name__ == "__main__":
    main()
