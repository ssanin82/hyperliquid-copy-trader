"""
L2BookRecorder - Records L2 order book for a specific token until sample size is reached.

Records all L2 book updates for the specified token (defaults to TOKEN)
and stops when L2BOOK_SAMPLE_SIZE is reached.
"""

# Constants
TOKEN = "ETH"
L2BOOK_SAMPLE_SIZE = 100000

import json
import time
import signal
import sys
import os
from datetime import datetime, UTC
from typing import Optional, Dict, Any
import websocket
import threading


class L2BookRecorder:
    """Records L2 order book for a specific symbol until sample size is reached."""
    
    MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
    
    def __init__(self, token: str = None, output_file: str = None, testnet: bool = False):
        """
        Initialize the L2 book recorder.
        
        Args:
            token: Token to record L2 book for (defaults to TOKEN)
            output_file: File to write L2 book to (defaults to f"l2Book_{TOKEN}.log")
            testnet: Use testnet endpoints
        """
        self.token = token if token is not None else TOKEN
        # Create sample_data directory if it doesn't exist
        os.makedirs("sample_data", exist_ok=True)
        default_file = os.path.join("sample_data", f"l2Book_{self.token}.log")
        self.output_file = output_file if output_file is not None else default_file
        self.testnet = testnet
        self.ws_url = self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL
        
        self.ws = None
        self.ws_thread = None
        self.running = False
        self.l2book_count = 0
        self.file_handle = None
        
    def _record_l2book(self, l2book: Dict[str, Any]):
        """Record an L2 book update to the output file."""
        try:
            # Format L2 book with timestamp
            l2book_record = {
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "unix_timestamp": time.time(),
                "l2book_number": self.l2book_count + 1,
                "data": l2book
            }
            
            # Write as JSON line
            self.file_handle.write(json.dumps(l2book_record, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
            self.l2book_count += 1
            
            # Print progress every 100 L2 book updates
            if self.l2book_count % 10 == 0:
                print(f"Recorded {self.l2book_count}/{L2BOOK_SAMPLE_SIZE} L2 book updates...")
            
            # Check if we've reached the sample size
            if self.l2book_count >= L2BOOK_SAMPLE_SIZE:
                print(f"\nReached sample size of {L2BOOK_SAMPLE_SIZE} L2 book updates. Stopping...")
                self.stop()
                
        except Exception as e:
            print(f"Error recording L2 book: {e}")
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle L2 book messages
            if "channel" in data and data["channel"] == "l2Book":
                if "data" in data:
                    l2book_data = data["data"]
                    # Check if this L2 book is for our token
                    l2book_coin = l2book_data.get("coin", "")
                    if l2book_coin == self.token:
                        if not self.running:
                            return
                        self._record_l2book(l2book_data)
            
            # Also handle direct L2 book data
            elif "data" in data and isinstance(data["data"], dict):
                l2book_data = data["data"]
                if "coin" in l2book_data:
                    l2book_coin = l2book_data.get("coin", "")
                    if l2book_coin == self.token:
                        if not self.running:
                            return
                        self._record_l2book(l2book_data)
                            
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
        if self.running and self.l2book_count < L2BOOK_SAMPLE_SIZE:
            time.sleep(2)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        print(f"WebSocket connected to Hyperliquid")
        print(f"Subscribing to L2 book for {self.token}...")
        
        try:
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "l2Book", "coin": self.token}
            }
            ws.send(json.dumps(subscribe_msg))
            print(f"Subscribed to L2 book for {self.token}. Waiting for L2 book updates...")
        except Exception as e:
            print(f"Error subscribing to L2 book: {e}")
    
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
        """Start recording L2 book."""
        # Open output file
        try:
            self.file_handle = open(self.output_file, 'w')
            
            # Write header
            header = {
                "type": "header",
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "token": self.token,
                "sample_size": L2BOOK_SAMPLE_SIZE,
                "description": f"Hyperliquid L2 book log for {self.token} - JSON Lines format"
            }
            self.file_handle.write(json.dumps(header, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
        except Exception as e:
            print(f"Error opening output file: {e}")
            return
        
        print("=" * 80)
        print("L2 BOOK RECORDER")
        print("=" * 80)
        print(f"Token: {self.token}")
        print(f"Output file: {self.output_file}")
        print(f"Sample size: {L2BOOK_SAMPLE_SIZE} L2 book updates")
        print(f"Recording all L2 book updates for {self.token}...")
        print("Press Ctrl+C to stop early")
        print("=" * 80)
        
        self.running = True
        
        # Start WebSocket
        self._start_websocket()
        
        try:
            # Keep running until stopped or sample size reached
            while self.running and self.l2book_count < L2BOOK_SAMPLE_SIZE:
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
                "total_l2book_recorded": self.l2book_count,
                "sample_size": L2BOOK_SAMPLE_SIZE,
                "completed": self.l2book_count >= L2BOOK_SAMPLE_SIZE
            }
            self.file_handle.write(json.dumps(footer, separators=(',', ':')) + "\n")
            self.file_handle.close()
            self.file_handle = None
        
        print(f"Recorded {self.l2book_count} L2 book updates")
        print(f"Output saved to: {self.output_file}")


def main():
    """Main entry point."""
    # Set up signal handler
    recorder = L2BookRecorder()
    
    def signal_handler(sig, frame):
        recorder.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start recording
    recorder.start()


if __name__ == "__main__":
    main()
