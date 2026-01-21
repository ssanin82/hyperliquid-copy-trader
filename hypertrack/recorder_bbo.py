"""
BBORecorder - Records BBO (Best Bid Offer) for a specific token until sample size is reached.

Records all BBO updates for the specified token (defaults to TOKEN)
and stops when BBO_SAMPLE_SIZE is reached.
"""

# Constants
TOKEN = "ETH"
BBO_SAMPLE_SIZE = 100000

import json
import time
import signal
import sys
from datetime import datetime, UTC
from typing import Optional, Dict, Any
import websocket
import threading


class BBORecorder:
    """Records BBO (Best Bid Offer) for a specific symbol until sample size is reached."""
    
    MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
    
    def __init__(self, token: str = None, output_file: str = None, testnet: bool = False):
        """
        Initialize the BBO recorder.
        
        Args:
            token: Token to record BBO for (defaults to TOKEN)
            output_file: File to write BBO to (defaults to f"bbo_{TOKEN}.log")
            testnet: Use testnet endpoints
        """
        self.token = token if token is not None else TOKEN
        self.output_file = output_file if output_file is not None else f"bbo_{self.token}.log"
        self.testnet = testnet
        self.ws_url = self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL
        
        self.ws = None
        self.ws_thread = None
        self.running = False
        self.bbo_count = 0
        self.file_handle = None
        
    def _record_bbo(self, bbo: Dict[str, Any]):
        """Record a BBO update to the output file."""
        try:
            # Format BBO with timestamp
            bbo_record = {
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "unix_timestamp": time.time(),
                "bbo_number": self.bbo_count + 1,
                "data": bbo
            }
            
            # Write as JSON line
            self.file_handle.write(json.dumps(bbo_record, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
            self.bbo_count += 1
            
            # Print progress every 100 BBO updates
            if self.bbo_count % 10 == 0:
                print(f"Recorded {self.bbo_count}/{BBO_SAMPLE_SIZE} BBO updates...")
            
            # Check if we've reached the sample size
            if self.bbo_count >= BBO_SAMPLE_SIZE:
                print(f"\nReached sample size of {BBO_SAMPLE_SIZE} BBO updates. Stopping...")
                self.stop()
                
        except Exception as e:
            print(f"Error recording BBO: {e}")
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle BBO messages
            if "channel" in data and data["channel"] == "bbo":
                if "data" in data:
                    bbo_data = data["data"]
                    # Check if this BBO is for our token
                    bbo_coin = bbo_data.get("coin", "")
                    if bbo_coin == self.token:
                        if not self.running:
                            return
                        self._record_bbo(bbo_data)
            
            # Also handle direct BBO data
            elif "data" in data and isinstance(data["data"], dict):
                bbo_data = data["data"]
                if "coin" in bbo_data:
                    bbo_coin = bbo_data.get("coin", "")
                    if bbo_coin == self.token:
                        if not self.running:
                            return
                        self._record_bbo(bbo_data)
                            
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
        if self.running and self.bbo_count < BBO_SAMPLE_SIZE:
            time.sleep(2)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        print(f"WebSocket connected to Hyperliquid")
        print(f"Subscribing to BBO for {self.token}...")
        
        try:
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "bbo", "coin": self.token}
            }
            ws.send(json.dumps(subscribe_msg))
            print(f"Subscribed to BBO for {self.token}. Waiting for BBO updates...")
        except Exception as e:
            print(f"Error subscribing to BBO: {e}")
    
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
        """Start recording BBO."""
        # Open output file
        try:
            self.file_handle = open(self.output_file, 'w')
            
            # Write header
            header = {
                "type": "header",
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "token": self.token,
                "sample_size": BBO_SAMPLE_SIZE,
                "description": f"Hyperliquid BBO log for {self.token} - JSON Lines format"
            }
            self.file_handle.write(json.dumps(header, separators=(',', ':')) + "\n")
            self.file_handle.flush()
            
        except Exception as e:
            print(f"Error opening output file: {e}")
            return
        
        print("=" * 80)
        print("BBO RECORDER")
        print("=" * 80)
        print(f"Token: {self.token}")
        print(f"Output file: {self.output_file}")
        print(f"Sample size: {BBO_SAMPLE_SIZE} BBO updates")
        print(f"Recording all BBO updates for {self.token}...")
        print("Press Ctrl+C to stop early")
        print("=" * 80)
        
        self.running = True
        
        # Start WebSocket
        self._start_websocket()
        
        try:
            # Keep running until stopped or sample size reached
            while self.running and self.bbo_count < BBO_SAMPLE_SIZE:
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
                "total_bbo_recorded": self.bbo_count,
                "sample_size": BBO_SAMPLE_SIZE,
                "completed": self.bbo_count >= BBO_SAMPLE_SIZE
            }
            self.file_handle.write(json.dumps(footer, separators=(',', ':')) + "\n")
            self.file_handle.close()
            self.file_handle = None
        
        print(f"Recorded {self.bbo_count} BBO updates")
        print(f"Output saved to: {self.output_file}")


def main():
    """Main entry point."""
    # Set up signal handler
    recorder = BBORecorder()
    
    def signal_handler(sig, frame):
        recorder.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start recording
    recorder.start()


if __name__ == "__main__":
    main()
