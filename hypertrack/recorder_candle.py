"""
CandleRecorder - Records candle (OHLCV) data for a specific token until sample size is reached.

Records all candle updates for the specified token (defaults to TOKEN)
and stops when CANDLE_SAMPLE_SIZE is reached.
"""

# Constants
TOKEN = "ETH"
CANDLE_SAMPLE_SIZE = 100000
CANDLE_INTERVAL = "1m"  # Supported: "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"

import json
import time
import signal
import sys
from datetime import datetime, UTC
from typing import Optional, Dict, Any
import websocket
import threading


class CandleRecorder:
    """Records candle (OHLCV) data for a specific symbol until sample size is reached."""
    
    MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
    
    def __init__(self, token: str = None, output_file: str = None, interval: str = None, testnet: bool = False):
        """
        Initialize the candle recorder.
        
        Args:
            token: Token to record candles for (defaults to TOKEN)
            output_file: File to write candles to (defaults to f"candle_{TOKEN}.log")
            interval: Candle interval (defaults to CANDLE_INTERVAL)
            testnet: Use testnet endpoints
        """
        self.token = token if token is not None else TOKEN
        self.output_file = output_file if output_file is not None else f"candle_{self.token}.log"
        self.interval = interval if interval is not None else CANDLE_INTERVAL
        self.testnet = testnet
        self.ws_url = self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL
        
        self.ws = None
        self.ws_thread = None
        self.running = False
        self.candle_count = 0
        self.file_handle = None
        
    def _record_candle(self, candle: Dict[str, Any]):
        """Record a candle update to the output file."""
        try:
            # Format candle with timestamp
            candle_record = {
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "unix_timestamp": time.time(),
                "candle_number": self.candle_count + 1,
                "data": candle
            }
            
            # Write as JSON line
            self.file_handle.write(json.dumps(candle_record, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
            self.candle_count += 1
            
            # Print progress every 100 candles
            if self.candle_count % 10 == 0:
                print(f"Recorded {self.candle_count}/{CANDLE_SAMPLE_SIZE} candles...")
            
            # Check if we've reached the sample size
            if self.candle_count >= CANDLE_SAMPLE_SIZE:
                print(f"\nReached sample size of {CANDLE_SAMPLE_SIZE} candles. Stopping...")
                self.stop()
                
        except Exception as e:
            print(f"Error recording candle: {e}")
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle candle messages
            if "channel" in data and data["channel"] == "candle":
                if "data" in data:
                    candle_data = data["data"]
                    # Check if this candle is for our token
                    # Candle data might be a list or dict, handle both
                    if isinstance(candle_data, list):
                        for candle in candle_data:
                            candle_coin = candle.get("coin", "") if isinstance(candle, dict) else ""
                            if candle_coin == self.token:
                                if not self.running:
                                    return
                                self._record_candle(candle)
                    elif isinstance(candle_data, dict):
                        candle_coin = candle_data.get("coin", "")
                        if candle_coin == self.token:
                            if not self.running:
                                return
                            self._record_candle(candle_data)
            
            # Also handle direct candle data
            elif "data" in data:
                candle_data = data["data"]
                if isinstance(candle_data, list):
                    for candle in candle_data:
                        if isinstance(candle, dict) and "coin" in candle:
                            candle_coin = candle.get("coin", "")
                            if candle_coin == self.token:
                                if not self.running:
                                    return
                                self._record_candle(candle)
                elif isinstance(candle_data, dict) and "coin" in candle_data:
                    candle_coin = candle_data.get("coin", "")
                    if candle_coin == self.token:
                        if not self.running:
                            return
                        self._record_candle(candle_data)
                            
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
        if self.running and self.candle_count < CANDLE_SAMPLE_SIZE:
            time.sleep(2)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        print(f"WebSocket connected to Hyperliquid")
        print(f"Subscribing to candles for {self.token} (interval: {self.interval})...")
        
        try:
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "candle", "coin": self.token, "interval": self.interval}
            }
            subscribe_json = json.dumps(subscribe_msg)
            print(f"Subscription message: {subscribe_json}")
            ws.send(subscribe_json)
            print(f"Sent subscription request. Waiting for candle updates...")
        except Exception as e:
            print(f"Error subscribing to candles: {e}")
    
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
        """Start recording candles."""
        # Open output file
        try:
            self.file_handle = open(self.output_file, 'w')
            
            # Write header
            header = {
                "type": "header",
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "token": self.token,
                "interval": self.interval,
                "sample_size": CANDLE_SAMPLE_SIZE,
                "description": f"Hyperliquid candle log for {self.token} ({self.interval}) - JSON Lines format"
            }
            self.file_handle.write(json.dumps(header, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
        except Exception as e:
            print(f"Error opening output file: {e}")
            return
        
        print("=" * 80)
        print("CANDLE RECORDER")
        print("=" * 80)
        print(f"Token: {self.token}")
        print(f"Interval: {self.interval}")
        print(f"Output file: {self.output_file}")
        print(f"Sample size: {CANDLE_SAMPLE_SIZE} candles")
        print(f"Recording all candles for {self.token}...")
        print("Press Ctrl+C to stop early")
        print("=" * 80)
        
        self.running = True
        
        # Start WebSocket
        self._start_websocket()
        
        try:
            # Keep running until stopped or sample size reached
            while self.running and self.candle_count < CANDLE_SAMPLE_SIZE:
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
                "total_candles_recorded": self.candle_count,
                "sample_size": CANDLE_SAMPLE_SIZE,
                "completed": self.candle_count >= CANDLE_SAMPLE_SIZE
            }
            self.file_handle.write(json.dumps(footer, separators=(',', ':')) + "\n")
            self.file_handle.close()
            self.file_handle = None
        
        print(f"Recorded {self.candle_count} candles")
        print(f"Output saved to: {self.output_file}")


def main():
    """Main entry point."""
    # Set up signal handler
    recorder = CandleRecorder()
    
    def signal_handler(sig, frame):
        recorder.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start recording
    recorder.start()


if __name__ == "__main__":
    main()
