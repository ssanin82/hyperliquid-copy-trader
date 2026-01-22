"""
Data Recorder - Stage 1: Records blockchain and API data.

Subscribes to:
- Blockchain events (blocks, transactions)
- Public API (trades, BBO, L2Book)

Saves all data to JSON Lines files and tracks metrics.
"""

import json
import time
import signal
import sys
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
import multiprocessing
from multiprocessing import Process, Queue as MPQueue, Event
import os

# Constants
MAINNET_RPC_HTTP = "https://hyperliquid-mainnet.g.alchemy.com/v2/E45W-MsgmrM0Ye2gH8ZoX"
MAINNET_RPC_WS = "wss://hyperliquid-mainnet.g.alchemy.com/v2/E45W-MsgmrM0Ye2gH8ZoX"
HYPERLIQUID_API_WS = "wss://api.hyperliquid.xyz/ws"

API_TOKENS = [
    "ETH",       # Ethereum
    # Add more tokens as needed
]

# Output directory for recorded data
DATA_DIR = "recorded_data"
BLOCKS_FILE = os.path.join(DATA_DIR, "blocks.jsonl")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.jsonl")
TRADES_FILE = os.path.join(DATA_DIR, "trades.jsonl")
BBO_FILE = os.path.join(DATA_DIR, "bbo.jsonl")
L2BOOK_FILE = os.path.join(DATA_DIR, "l2book.jsonl")
METRICS_FILE = os.path.join(DATA_DIR, "data_collection_report.txt")


def blockchain_worker(stop_event: Event, output_queue: MPQueue, rpc_url: str):
    """Worker process for blockchain subscription."""
    try:
        import websocket
        import requests
        
        reconnect_delay = 2
        max_reconnect_delay = 60
        
        def json_rpc_request(method: str, params: list) -> Optional[Dict[str, Any]]:
            """Make JSON-RPC request."""
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1
            }
            try:
                http_url = rpc_url.replace("wss://", "https://").replace("ws://", "http://")
                response = requests.post(
                    http_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        return None
                    return result.get("result")
            except Exception:
                return None
        
        def on_message(ws, message):
            """Handle WebSocket messages."""
            if stop_event.is_set():
                ws.close()
                return
            
            try:
                data = json.loads(message)
                
                if "id" in data and "result" in data:
                    # Subscription confirmed
                    return
                
                if data.get("method") == "eth_subscription" and "params" in data:
                    params = data.get("params", {})
                    result = params.get("result")
                    if result:
                        block_number = result.get("number")
                        if block_number:
                            # Get full block
                            block = json_rpc_request("eth_getBlockByNumber", [block_number, True])
                            if block:
                                # Send to main process
                                output_queue.put({
                                    "type": "block",
                                    "data": block
                                })
            except Exception as e:
                output_queue.put({
                    "type": "error",
                    "source": "blockchain",
                    "error": str(e)
                })
        
        def on_open(ws):
            """Handle WebSocket open."""
            nonlocal reconnect_delay
            reconnect_delay = 2
            
            subscribe_msg = {
                "jsonrpc": "2.0",
                "method": "eth_subscribe",
                "params": ["newHeads"],
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))
        
        def on_error(ws, error):
            """Handle WebSocket error."""
            if not stop_event.is_set():
                output_queue.put({
                    "type": "error",
                    "source": "blockchain",
                    "error": str(error)
                })
        
        def on_close(ws, close_status_code, close_msg):
            """Handle WebSocket close."""
            pass
        
        # Main loop with reconnection
        while not stop_event.is_set():
            try:
                ws = websocket.WebSocketApp(
                    rpc_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except KeyboardInterrupt:
                break
            except Exception as e:
                if not stop_event.is_set():
                    output_queue.put({
                        "type": "error",
                        "source": "blockchain",
                        "error": str(e)
                    })
            
            # Reconnect with exponential backoff
            if not stop_event.is_set():
                sleep_chunk = 0.5
                slept = 0.0
                while slept < reconnect_delay and not stop_event.is_set():
                    try:
                        time.sleep(min(sleep_chunk, reconnect_delay - slept))
                        slept += sleep_chunk
                    except KeyboardInterrupt:
                        break
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    except KeyboardInterrupt:
        pass


def api_subscription_worker(stop_event: Event, output_queue: MPQueue, channel: str, coins: List[str]):
    """Worker process for Hyperliquid API subscription."""
    try:
        import websocket
        
        reconnect_delay = 2
        max_reconnect_delay = 60
        
        def on_message(ws, message):
            """Handle WebSocket messages."""
            if stop_event.is_set():
                ws.close()
                return
            
            try:
                data = json.loads(message)
                message_channel = data.get("channel", "")
                
                # Send to main process if it's a data message
                if message_channel != "subscriptionResponse":
                    output_queue.put({
                        "type": "api_data",
                        "channel": channel,
                        "data": data
                    })
            except Exception as e:
                if not stop_event.is_set():
                    output_queue.put({
                        "type": "error",
                        "source": channel,
                        "error": str(e)
                    })
        
        def on_open(ws):
            """Handle WebSocket open."""
            nonlocal reconnect_delay
            reconnect_delay = 2
            
            try:
                for coin in coins:
                    subscribe_msg = {"method": "subscribe", "subscription": {"type": channel, "coin": coin}}
                    try:
                        ws.send(json.dumps(subscribe_msg))
                        if not stop_event.is_set():
                            output_queue.put({
                                "type": "subscription_made",
                                "source": channel,
                                "coin": coin
                            })
                        time.sleep(0.1)
                    except Exception as e:
                        if not stop_event.is_set():
                            output_queue.put({
                                "type": "error",
                                "source": channel,
                                "error": f"Error sending subscription for {coin}: {str(e)}"
                            })
            except Exception as e:
                if not stop_event.is_set():
                    output_queue.put({
                        "type": "error",
                        "source": channel,
                        "error": f"Error during subscription: {str(e)}"
                    })
        
        def on_error(ws, error):
            """Handle WebSocket error."""
            if not stop_event.is_set():
                output_queue.put({
                    "type": "error",
                    "source": channel,
                    "error": str(error)
                })
        
        def on_close(ws, close_status_code, close_msg):
            """Handle WebSocket close."""
            pass
        
        # Main loop with reconnection
        while not stop_event.is_set():
            try:
                ws_instance = websocket.WebSocketApp(
                    HYPERLIQUID_API_WS,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                ws_instance.run_forever(ping_interval=30, ping_timeout=10)
            except KeyboardInterrupt:
                break
            except Exception as e:
                if not stop_event.is_set():
                    output_queue.put({
                        "type": "error",
                        "source": channel,
                        "error": str(e)
                    })
            
            # Reconnect with exponential backoff
            if not stop_event.is_set():
                sleep_chunk = 0.5
                slept = 0.0
                while slept < reconnect_delay and not stop_event.is_set():
                    try:
                        time.sleep(min(sleep_chunk, reconnect_delay - slept))
                        slept += sleep_chunk
                    except KeyboardInterrupt:
                        break
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    except KeyboardInterrupt:
        pass


class DataRecorder:
    """Records blockchain and API data to files."""
    
    def __init__(self, data_dir: str = DATA_DIR):
        """Initialize the data recorder."""
        self.data_dir = data_dir
        self.running = False
        self.stop_event = Event()
        self.processes = []
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Open output files
        self.blocks_file = open(BLOCKS_FILE, 'w')
        self.transactions_file = open(TRANSACTIONS_FILE, 'w')
        self.trades_file = open(TRADES_FILE, 'w')
        self.bbo_file = open(BBO_FILE, 'w')
        self.l2book_file = open(L2BOOK_FILE, 'w')
        
        # Metrics
        self.metrics = {
            'blocks_received': 0,
            'transactions_received': 0,
            'trades_received': 0,
            'bbo_received': 0,
            'l2book_received': 0,
            'start_time': None,  # ISO string
            'start_timestamp': None,  # Unix timestamp for calculations
            'end_time': None,
            'subscriptions_made': 0,
            'last_block_number': None  # Last recorded block number
        }
        
        # Queues
        self.blockchain_queue = MPQueue()
        self.trades_queue = MPQueue()
        self.bbo_queue = MPQueue()
        self.l2book_queue = MPQueue()
    
    def _write_jsonl(self, file_handle, data: Dict[str, Any]):
        """Write a JSON line to file."""
        record = {
            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "unix_timestamp": time.time(),
            "data": data
        }
        file_handle.write(json.dumps(record, separators=(',', ':')) + "\n")
        file_handle.flush()
    
    def _process_blockchain_queue(self):
        """Process messages from blockchain queue."""
        while not self.blockchain_queue.empty():
            try:
                msg = self.blockchain_queue.get_nowait()
                if msg.get("type") == "block":
                    block = msg.get("data")
                    if block:
                        self._write_jsonl(self.blocks_file, block)
                        self.metrics['blocks_received'] += 1
                        
                        # Track last block number (keep in hex format as from chain)
                        block_number = block.get("number")
                        if block_number:
                            # Keep as hex string if it's already hex, otherwise convert to hex
                            if isinstance(block_number, str):
                                # Already in hex format (e.g., "0x180dab8")
                                if block_number.startswith("0x"):
                                    self.metrics['last_block_number'] = block_number
                                else:
                                    # Try to convert decimal string to hex
                                    try:
                                        num = int(block_number)
                                        self.metrics['last_block_number'] = hex(num)
                                    except ValueError:
                                        pass
                            elif isinstance(block_number, int):
                                # Convert int to hex
                                self.metrics['last_block_number'] = hex(block_number)
                        
                        # Extract and save transactions
                        transactions = block.get("transactions", [])
                        for tx in transactions:
                            self._write_jsonl(self.transactions_file, tx)
                            self.metrics['transactions_received'] += 1
                elif msg.get("type") == "error":
                    print(f"[ERROR] Blockchain: {msg.get('error')}")
            except:
                break
    
    def _process_api_queue(self, queue, file_handle, metric_key, channel_name):
        """Process messages from an API queue."""
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                if msg.get("type") == "api_data":
                    data = msg.get("data", {})
                    message_channel = data.get("channel", "")
                    if message_channel == channel_name:
                        api_data = data.get("data", {})
                        if api_data:
                            self._write_jsonl(file_handle, api_data)
                            self.metrics[metric_key] += 1
                elif msg.get("type") == "subscription_made":
                    self.metrics['subscriptions_made'] += 1
                elif msg.get("type") == "error":
                    print(f"[ERROR] {channel_name}: {msg.get('error')}")
            except:
                break
    
    def _start_processes(self):
        """Start all subscription processes."""
        # Start blockchain process
        p = Process(target=blockchain_worker, args=(self.stop_event, self.blockchain_queue, MAINNET_RPC_WS))
        p.start()
        self.processes.append(p)
        print("Started blockchain subscription process")
        
        # Start API subscription processes
        queue_map = {
            "trades": (self.trades_queue, self.trades_file, "trades_received", "trades"),
            "bbo": (self.bbo_queue, self.bbo_file, "bbo_received", "bbo"),
            "l2Book": (self.l2book_queue, self.l2book_file, "l2book_received", "l2Book")
        }
        
        for channel, (queue, file_handle, metric_key, channel_name) in queue_map.items():
            p = Process(target=api_subscription_worker, args=(self.stop_event, queue, channel, API_TOKENS))
            p.start()
            self.processes.append(p)
            print(f"Started {channel} subscription process")
    
    def _generate_report(self):
        """Generate data collection report."""
        self.metrics['end_time'] = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        duration = time.time() - self.metrics['start_timestamp'] if self.metrics['start_timestamp'] else 0
        
        with open(METRICS_FILE, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA COLLECTION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {self.metrics['end_time']}\n")
            f.write(f"Start time: {self.metrics.get('start_time', 'N/A')}\n")
            f.write(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n\n")
            
            f.write("METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Blocks received: {self.metrics['blocks_received']}\n")
            f.write(f"Transactions received: {self.metrics['transactions_received']}\n")
            f.write(f"Trades received: {self.metrics['trades_received']}\n")
            f.write(f"BBO updates received: {self.metrics['bbo_received']}\n")
            f.write(f"L2Book updates received: {self.metrics['l2book_received']}\n")
            f.write(f"Total subscriptions made: {self.metrics['subscriptions_made']}\n\n")
            
            f.write("DATA FILES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Blocks: {BLOCKS_FILE}\n")
            f.write(f"Transactions: {TRANSACTIONS_FILE}\n")
            f.write(f"Trades: {TRADES_FILE}\n")
            f.write(f"BBO: {BBO_FILE}\n")
            f.write(f"L2Book: {L2BOOK_FILE}\n")
        
        print(f"\nData collection report saved to: {METRICS_FILE}")
    
    def start(self):
        """Start recording data."""
        self.running = True
        self.metrics['start_timestamp'] = time.time()
        self.metrics['start_time'] = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        
        print("=" * 80)
        print("DATA RECORDER - STAGE 1")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Recording blockchain and API data...")
        print("Press Ctrl+C to stop")
        print("=" * 80)
        
        # Start all processes
        self._start_processes()
        
        try:
            last_status = time.time()
            while self.running:
                # Process queues
                self._process_blockchain_queue()
                self._process_api_queue(self.trades_queue, self.trades_file, "trades_received", "trades")
                self._process_api_queue(self.bbo_queue, self.bbo_file, "bbo_received", "bbo")
                self._process_api_queue(self.l2book_queue, self.l2book_file, "l2book_received", "l2Book")
                
                # Print status every 10 seconds
                if time.time() - last_status > 10:
                    # Calculate elapsed time
                    elapsed = time.time() - self.metrics['start_timestamp'] if self.metrics['start_timestamp'] else 0
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)
                    
                    # Format time string - only show non-zero units
                    time_parts = []
                    if hours > 0:
                        time_parts.append(f"{hours}h")
                    if minutes > 0:
                        time_parts.append(f"{minutes}m")
                    time_parts.append(f"{seconds}s")
                    time_str = " ".join(time_parts)
                    
                    last_block_str = f"#{self.metrics['last_block_number']}" if self.metrics['last_block_number'] is not None else "#N/A"
                    print(f"[STATUS] Blocks: {self.metrics['blocks_received']}, last block {last_block_str}, "
                          f"TXs: {self.metrics['transactions_received']}, "
                          f"Trades: {self.metrics['trades_received']}, "
                          f"BBO: {self.metrics['bbo_received']}, "
                          f"L2Book: {self.metrics['l2book_received']}, "
                          f"time took: {time_str}")
                    last_status = time.time()
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop recording and generate report."""
        print("\nStopping data recorder...")
        self.running = False
        self.stop_event.set()
        
        # Wait for processes
        for p in self.processes:
            p.join(timeout=5)
        
        # Close files
        self.blocks_file.close()
        self.transactions_file.close()
        self.trades_file.close()
        self.bbo_file.close()
        self.l2book_file.close()
        
        # Generate report
        self._generate_report()
        print("Data recording stopped.")


def main():
    """Main entry point."""
    recorder = DataRecorder()
    
    def signal_handler(sig, frame):
        recorder.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    recorder.start()


if __name__ == "__main__":
    main()
