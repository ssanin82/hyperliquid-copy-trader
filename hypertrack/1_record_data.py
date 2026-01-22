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
    "HYPE",      # Hyperliquid native token
    "BTC",       # Bitcoin
    "SOL",       # Solana
    "AVAX",      # Avalanche
    "OP",        # Optimism
    "BNB",       # Binance Coin
    "DOGE",      # Dogecoin
    "LINK",      # Chainlink
    "MATIC",     # Polygon
    # "LTC",       # Litecoin
    # "ARB",       # Arbitrum
    # "FTM",       # Fantom
    # "XRP",       # Ripple
    # "ETC",       # Ethereum Classic
    # "APE",       # ApeCoin
    # "PUMP",      # Pump token (high volume meme token) 
    # "PURR",      # Hypurr Fun / PURR
    # "KNTQ",      # Kinetiq
    # "SEDA"       # SEDA Protocol token
]

RECORD_BLOCKS = True
RECORD_TRANSACTIONS = True
RECORD_TRADES = True
RECORD_BBO = True
RECORD_L2BOOK = True
RECORD_CANDLES = False

# Candle interval
CANDLE_INTERVAL = "1m"  # Supported: "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"

# Output directory for recorded data
DATA_DIR = "recorded_data"
BLOCKS_FILE = os.path.join(DATA_DIR, "blocks.jsonl")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.jsonl")
TRADES_FILE = os.path.join(DATA_DIR, "trades.jsonl")
BBO_FILE = os.path.join(DATA_DIR, "bbo.jsonl")
L2BOOK_FILE = os.path.join(DATA_DIR, "l2book.jsonl")
CANDLES_FILE = os.path.join(DATA_DIR, "candles.jsonl")
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


def candle_subscription_worker(stop_event: Event, output_queue: MPQueue, coins: List[str], interval: str):
    """Worker process for Hyperliquid candle subscription."""
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
                
                # Handle subscription response
                if message_channel == "subscriptionResponse":
                    return
                
                # Handle candle messages
                if message_channel == "candle" and "data" in data:
                    candle_data = data["data"]
                    
                    # Candle data can be a list or dict
                    # Candle data uses "s" field for symbol/coin
                    candles_to_send = []
                    
                    if isinstance(candle_data, list):
                        for candle in candle_data:
                            if isinstance(candle, dict):
                                # Check both "s" (symbol) and "coin" fields
                                candle_coin = candle.get("s", candle.get("coin", ""))
                                if candle_coin in coins:
                                    candles_to_send.append(candle)
                    elif isinstance(candle_data, dict):
                        # Check both "s" (symbol) and "coin" fields
                        candle_coin = candle_data.get("s", candle_data.get("coin", ""))
                        if candle_coin in coins:
                            candles_to_send.append(candle_data)
                    
                    # Send each candle to main process
                    for candle in candles_to_send:
                        output_queue.put({
                            "type": "api_data",
                            "channel": "candle",
                            "data": {"channel": "candle", "data": candle}
                        })
            except Exception as e:
                if not stop_event.is_set():
                    output_queue.put({
                        "type": "error",
                        "source": "candle",
                        "error": str(e)
                    })
        
        def on_open(ws):
            """Handle WebSocket open."""
            nonlocal reconnect_delay
            reconnect_delay = 2
            
            try:
                for coin in coins:
                    subscribe_msg = {
                        "method": "subscribe",
                        "subscription": {"type": "candle", "coin": coin, "interval": interval}
                    }
                    try:
                        ws.send(json.dumps(subscribe_msg))
                        if not stop_event.is_set():
                            output_queue.put({
                                "type": "subscription_made",
                                "source": "candle",
                                "coin": coin
                            })
                        time.sleep(0.1)
                    except Exception as e:
                        if not stop_event.is_set():
                            output_queue.put({
                                "type": "error",
                                "source": "candle",
                                "error": f"Error sending subscription for {coin}: {str(e)}"
                            })
            except Exception as e:
                if not stop_event.is_set():
                    output_queue.put({
                        "type": "error",
                        "source": "candle",
                        "error": f"Error during subscription: {str(e)}"
                    })
        
        def on_error(ws, error):
            """Handle WebSocket error."""
            if not stop_event.is_set():
                output_queue.put({
                    "type": "error",
                    "source": "candle",
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
                        "source": "candle",
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
        
        # Clear existing files in the data directory
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
        
        # Open output files conditionally
        self.blocks_file = open(BLOCKS_FILE, 'w') if RECORD_BLOCKS else None
        self.transactions_file = open(TRANSACTIONS_FILE, 'w') if RECORD_TRANSACTIONS else None
        self.trades_file = open(TRADES_FILE, 'w') if RECORD_TRADES else None
        self.bbo_file = open(BBO_FILE, 'w') if RECORD_BBO else None
        self.l2book_file = open(L2BOOK_FILE, 'w') if RECORD_L2BOOK else None
        self.candles_file = open(CANDLES_FILE, 'w') if RECORD_CANDLES else None
        
        # Metrics
        self.metrics = {
            'blocks_received': 0,
            'transactions_received': 0,
            'trades_received': 0,
            'bbo_received': 0,
            'l2book_received': 0,
            'candles_received': 0,
            'start_time': None,  # ISO string
            'start_timestamp': None,  # Unix timestamp for calculations
            'end_time': None,
            'subscriptions_made': 0,
            'last_block_number': None  # Last recorded block number
        }
        
        # Queues (only create if needed)
        self.blockchain_queue = MPQueue() if RECORD_BLOCKS or RECORD_TRANSACTIONS else None
        self.trades_queue = MPQueue() if RECORD_TRADES else None
        self.bbo_queue = MPQueue() if RECORD_BBO else None
        self.l2book_queue = MPQueue() if RECORD_L2BOOK else None
        self.candles_queue = MPQueue() if RECORD_CANDLES else None
    
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
        if not RECORD_BLOCKS and not RECORD_TRANSACTIONS:
            return
        if self.blockchain_queue is None:
            return
            
        while not self.blockchain_queue.empty():
            try:
                msg = self.blockchain_queue.get_nowait()
                if msg.get("type") == "block":
                    block = msg.get("data")
                    if block:
                        if RECORD_BLOCKS and self.blocks_file:
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
                        if RECORD_TRANSACTIONS and self.transactions_file:
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
        if queue is None or file_handle is None:
            return
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
        # Start blockchain process (if blocks or transactions are enabled)
        if RECORD_BLOCKS or RECORD_TRANSACTIONS:
            if self.blockchain_queue:
                p = Process(target=blockchain_worker, args=(self.stop_event, self.blockchain_queue, MAINNET_RPC_WS))
                p.start()
                self.processes.append(p)
                print("Started blockchain subscription process")
        
        # Start API subscription processes
        queue_map = {
            "trades": (RECORD_TRADES, self.trades_queue, self.trades_file, "trades_received", "trades"),
            "bbo": (RECORD_BBO, self.bbo_queue, self.bbo_file, "bbo_received", "bbo"),
            "l2Book": (RECORD_L2BOOK, self.l2book_queue, self.l2book_file, "l2book_received", "l2Book")
        }
        
        for channel, (enabled, queue, file_handle, metric_key, channel_name) in queue_map.items():
            if enabled and queue:
                p = Process(target=api_subscription_worker, args=(self.stop_event, queue, channel, API_TOKENS))
                p.start()
                self.processes.append(p)
                print(f"Started {channel} subscription process")
        
        # Start candle subscription process
        if RECORD_CANDLES and self.candles_queue:
            p = Process(target=candle_subscription_worker, args=(self.stop_event, self.candles_queue, API_TOKENS, CANDLE_INTERVAL))
            p.start()
            self.processes.append(p)
            print(f"Started candle subscription process (interval: {CANDLE_INTERVAL})")
    
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
            if RECORD_BLOCKS:
                f.write(f"Blocks received: {self.metrics['blocks_received']}\n")
            if RECORD_TRANSACTIONS:
                f.write(f"Transactions received: {self.metrics['transactions_received']}\n")
            if RECORD_TRADES:
                f.write(f"Trades received: {self.metrics['trades_received']}\n")
            if RECORD_BBO:
                f.write(f"BBO updates received: {self.metrics['bbo_received']}\n")
            if RECORD_L2BOOK:
                f.write(f"L2Book updates received: {self.metrics['l2book_received']}\n")
            if RECORD_CANDLES:
                f.write(f"Candles received: {self.metrics['candles_received']}\n")
            f.write(f"Total subscriptions made: {self.metrics['subscriptions_made']}\n\n")
            
            f.write("DATA FILES:\n")
            f.write("-" * 80 + "\n")
            if RECORD_BLOCKS:
                f.write(f"Blocks: {BLOCKS_FILE}\n")
            if RECORD_TRANSACTIONS:
                f.write(f"Transactions: {TRANSACTIONS_FILE}\n")
            if RECORD_TRADES:
                f.write(f"Trades: {TRADES_FILE}\n")
            if RECORD_BBO:
                f.write(f"BBO: {BBO_FILE}\n")
            if RECORD_L2BOOK:
                f.write(f"L2Book: {L2BOOK_FILE}\n")
            if RECORD_CANDLES:
                f.write(f"Candles: {CANDLES_FILE}\n")
        
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
                if RECORD_BLOCKS or RECORD_TRANSACTIONS:
                    self._process_blockchain_queue()
                if RECORD_TRADES:
                    self._process_api_queue(self.trades_queue, self.trades_file, "trades_received", "trades")
                if RECORD_BBO:
                    self._process_api_queue(self.bbo_queue, self.bbo_file, "bbo_received", "bbo")
                if RECORD_L2BOOK:
                    self._process_api_queue(self.l2book_queue, self.l2book_file, "l2book_received", "l2Book")
                if RECORD_CANDLES:
                    self._process_api_queue(self.candles_queue, self.candles_file, "candles_received", "candle")
                
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
                    
                    # Build status line with only enabled recorders
                    status_parts = []
                    if RECORD_BLOCKS:
                        last_block_str = f"#{self.metrics['last_block_number']}" if self.metrics['last_block_number'] is not None else "#N/A"
                        status_parts.append(f"Blocks: {self.metrics['blocks_received']}, last block {last_block_str}")
                    if RECORD_TRANSACTIONS:
                        status_parts.append(f"TXs: {self.metrics['transactions_received']}")
                    if RECORD_TRADES:
                        status_parts.append(f"Trades: {self.metrics['trades_received']}")
                    if RECORD_BBO:
                        status_parts.append(f"BBO: {self.metrics['bbo_received']}")
                    if RECORD_L2BOOK:
                        status_parts.append(f"L2Book: {self.metrics['l2book_received']}")
                    if RECORD_CANDLES:
                        status_parts.append(f"Candles: {self.metrics['candles_received']}")
                    
                    status_parts.append(f"time took: {time_str}")
                    print(f"[STATUS] {', '.join(status_parts)}")
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
        
        # Close files (only if they were opened)
        if self.blocks_file:
            self.blocks_file.close()
        if self.transactions_file:
            self.transactions_file.close()
        if self.trades_file:
            self.trades_file.close()
        if self.bbo_file:
            self.bbo_file.close()
        if self.l2book_file:
            self.l2book_file.close()
        if self.candles_file:
            self.candles_file.close()
        
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
