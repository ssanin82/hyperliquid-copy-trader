"""
ChainSubscriber - Subscribe to Hyperliquid on-chain block updates and log transactions.

Part of the hypertrack package. Connects to Hyperliquid's EVM-compatible blockchain RPC
and subscribes to new blocks, logging all transactions from each block.
"""

import json
import time
import signal
import sys
from datetime import datetime, UTC
from typing import Dict, Any, Optional
import requests
import websocket
import threading


class ChainSubscriber:
    """Subscribes to Hyperliquid blockchain block updates and logs transactions."""
    
    # Hyperliquid EVM-compatible RPC endpoints
    # Using Alchemy RPC for reliable EVM-compatible access
    MAINNET_RPC_HTTP = "https://hyperliquid-mainnet.g.alchemy.com/v2/E45W-MsgmrM0Ye2gH8ZoX"
    MAINNET_RPC_WS = "wss://hyperliquid-mainnet.g.alchemy.com/v2/E45W-MsgmrM0Ye2gH8ZoX"
    TESTNET_RPC_HTTP = "https://api.hyperliquid-testnet.xyz/info"
    TESTNET_RPC_WS = "wss://api.hyperliquid-testnet.xyz/ws"
    
    def __init__(self, testnet: bool = False, log_file: Optional[str] = None, rpc_url: Optional[str] = None):
        """
        Initialize the chain subscriber.
        
        Args:
            testnet: Use testnet endpoints
            log_file: Optional file to log transactions to (if None, only prints to console)
            rpc_url: Optional custom RPC URL (if provided, overrides default)
        """
        self.testnet = testnet
        if rpc_url:
            self.rpc_http = rpc_url.replace("wss://", "https://").replace("ws://", "http://")
            self.rpc_ws = rpc_url.replace("https://", "wss://").replace("http://", "ws://")
        else:
            self.rpc_http = self.TESTNET_RPC_HTTP if testnet else self.MAINNET_RPC_HTTP
            self.rpc_ws = self.TESTNET_RPC_WS if testnet else self.MAINNET_RPC_WS
        
        self.log_file = log_file
        self.file_handle = None
        self.running = False
        self.ws = None
        self.ws_thread = None
        self.transaction_count = 0
        self.last_block_number = None
        self.subscription_id = None
        self.eoa_addresses = set()  # Store unique EOA addresses
        self.checked_addresses = {}  # Cache: address -> is_eoa (True/False/None if not checked)
        
    def _log_transaction(self, tx_data: Dict[str, Any]):
        """Log a transaction to console and optionally to file."""
        self.transaction_count += 1
        
        # Format transaction log
        log_entry = {
            "transaction_number": self.transaction_count,
            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "unix_timestamp": time.time(),
            "data": tx_data
        }
        
        # Print to console
        print(f"\n[Transaction #{self.transaction_count}]")
        print(json.dumps(log_entry, indent=2))
        print("-" * 80)
        
        # Write to file if specified
        if self.file_handle:
            self.file_handle.write(json.dumps(log_entry, separators=(',', ':')) + "\n")
            self.file_handle.flush()
    
    def _json_rpc_request(self, method: str, params: list, request_id: int = 1) -> Dict[str, Any]:
        """Make a JSON-RPC request."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id
        }
        
        try:
            response = requests.post(
                self.rpc_http,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    print(f"JSON-RPC error: {result['error']}")
                    return None
                return result.get("result")
            else:
                print(f"HTTP error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error making JSON-RPC request: {e}")
            return None
    
    def _is_eoa(self, address: str) -> bool:
        """Check if an address is an EOA (Externally Owned Account) or a contract."""
        if not address or address == "0x" or address == "0x0":
            return False
        
        # Normalize address (lowercase)
        address = address.lower()
        
        # Check cache first
        if address in self.checked_addresses:
            return self.checked_addresses[address]
        
        # Use eth_getCode - if it returns "0x" or empty, it's an EOA
        code = self._json_rpc_request("eth_getCode", [address, "latest"])
        
        is_eoa = False
        if code is None:
            # Error or None, assume EOA for now
            is_eoa = True
        elif code == "0x" or code == "" or code == "0x0":
            # No code means it's an EOA
            is_eoa = True
        else:
            # Has code means it's a contract
            is_eoa = False
        
        # Cache the result
        self.checked_addresses[address] = is_eoa
        
        return is_eoa
    
    def _check_and_collect_eoa(self, address: str):
        """Check if address is EOA and add to collection if it is."""
        if not address or address == "0x" or address == "0x0":
            return
        
        address = address.lower()
        
        if self._is_eoa(address):
            self.eoa_addresses.add(address)
    
    def _get_block_by_number(self, block_number: str, full_transactions: bool = True) -> Optional[Dict[str, Any]]:
        """Get block by number using eth_getBlockByNumber."""
        return self._json_rpc_request("eth_getBlockByNumber", [block_number, full_transactions])
    
    def _get_latest_block_number(self) -> Optional[str]:
        """Get the latest block number."""
        result = self._json_rpc_request("eth_blockNumber", [])
        return result
    
    def _process_block(self, block: Dict[str, Any]):
        """Process a block and log all transactions."""
        if not block:
            return
        
        block_number = block.get("number")
        block_hash = block.get("hash")
        transactions = block.get("transactions", [])
        
        if block_number:
            # Convert hex to int for comparison
            try:
                current_block = int(block_number, 16) if isinstance(block_number, str) else block_number
                if self.last_block_number and current_block <= self.last_block_number:
                    return  # Already processed this block
                self.last_block_number = current_block
            except (ValueError, TypeError):
                pass
        
        # Process each transaction in the block (don't log block header)
        for tx in transactions:
            tx_data = None
            tx_from = None
            tx_to = None
            
            if isinstance(tx, dict):
                tx_data = tx
                tx_from = tx.get("from")
                tx_to = tx.get("to")
            elif isinstance(tx, str):
                # Transaction hash only, fetch full transaction
                tx_data = self._json_rpc_request("eth_getTransactionByHash", [tx])
                if tx_data:
                    tx_from = tx_data.get("from")
                    tx_to = tx_data.get("to")
            
            if tx_data:
                # Check and collect EOA addresses
                if tx_from:
                    self._check_and_collect_eoa(tx_from)
                if tx_to:
                    self._check_and_collect_eoa(tx_to)
                
                # Only log transactions (not blocks)
                self._log_transaction({
                    "type": "transaction",
                    "block_number": block_number,
                    "block_hash": block_hash,
                    "transaction": {
                        "hash": tx_data.get("hash"),
                        "from": tx_from,
                        "to": tx_to,
                        "value": tx_data.get("value"),
                        "gas": tx_data.get("gas"),
                        "gasPrice": tx_data.get("gasPrice"),
                        "input": tx_data.get("input"),
                        "nonce": tx_data.get("nonce"),
                        "transactionIndex": tx_data.get("transactionIndex"),
                    }
                })
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages (for eth_subscribe)."""
        try:
            data = json.loads(message)
            
            # Handle subscription confirmation
            if "id" in data and "result" in data:
                self.subscription_id = data["result"]
                print(f"Subscribed to new blocks. Subscription ID: {self.subscription_id}")
                return
            
            # Handle subscription notifications
            if data.get("method") == "eth_subscription" or "params" in data:
                params = data.get("params", {})
                subscription = params.get("subscription")
                result = params.get("result")
                
                if subscription == self.subscription_id and result:
                    # This is a new block header
                    block_number = result.get("number")
                    if block_number:
                        # Get full block with transactions
                        full_block = self._get_block_by_number(block_number, True)
                        if full_block:
                            self._process_block(full_block)
            
            # Also handle direct block data
            if "number" in data and "transactions" in data:
                self._process_block(data)
                    
        except json.JSONDecodeError as e:
            print(f"Error parsing WebSocket message: {e}")
            print(f"Raw message: {message[:200]}")
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_ws_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.running:
            print("Attempting to reconnect...")
            time.sleep(2)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open - subscribe to new blocks."""
        print("WebSocket connected to Hyperliquid blockchain RPC")
        print("Subscribing to new block headers...")
        
        try:
            # Subscribe to new block headers using eth_subscribe
            subscribe_msg = {
                "jsonrpc": "2.0",
                "method": "eth_subscribe",
                "params": ["newHeads"],
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))
            print("Subscription request sent. Waiting for confirmation...")
        except Exception as e:
            print(f"Error subscribing: {e}")
    
    def _poll_blocks(self):
        """Poll for new blocks (fallback method if WebSocket subscription fails)."""
        print("Starting block polling as fallback...")
        
        while self.running:
            try:
                # Get latest block number
                latest_block_hex = self._get_latest_block_number()
                if not latest_block_hex:
                    time.sleep(5)
                    continue
                
                try:
                    latest_block = int(latest_block_hex, 16) if isinstance(latest_block_hex, str) else latest_block_hex
                except (ValueError, TypeError):
                    time.sleep(5)
                    continue
                
                # Check if this is a new block
                if self.last_block_number is None:
                    # First time, start from current block
                    self.last_block_number = latest_block - 1
                
                # Process new blocks
                if latest_block > self.last_block_number:
                    for block_num in range(self.last_block_number + 1, latest_block + 1):
                        block_hex = hex(block_num)
                        full_block = self._get_block_by_number(block_hex, True)
                        if full_block:
                            self._process_block(full_block)
                        time.sleep(0.1)  # Small delay between blocks
                
                time.sleep(1)  # Poll every second
            except Exception as e:
                print(f"Error polling blocks: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    def _start_websocket(self):
        """Start WebSocket connection for eth_subscribe."""
        try:
            self.ws = websocket.WebSocketApp(
                self.rpc_ws,
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
            print("Falling back to HTTP polling...")
    
    def start(self):
        """Start subscribing to on-chain block updates."""
        # Open log file if specified
        if self.log_file:
            try:
                self.file_handle = open(self.log_file, 'w')
                header = {
                    "type": "header",
                    "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                    "description": "Hyperliquid on-chain block and transaction log - JSON Lines format",
                    "testnet": self.testnet,
                    "rpc_endpoint": self.rpc_http
                }
                self.file_handle.write(json.dumps(header, separators=(',', ':')) + "\n")
                self.file_handle.flush()
            except Exception as e:
                print(f"Error opening log file: {e}")
        
        print("=" * 80)
        print("HYPERLIQUID CHAIN SUBSCRIBER")
        print("=" * 80)
        print(f"Network: {'Testnet' if self.testnet else 'Mainnet'}")
        print(f"RPC HTTP: {self.rpc_http}")
        print(f"RPC WebSocket: {self.rpc_ws}")
        if self.log_file:
            print(f"Log file: {self.log_file}")
        print("Subscribing to new block updates...")
        print("Press Ctrl+C to stop")
        print("=" * 80)
        
        self.running = True
        
        # Start WebSocket connection for eth_subscribe
        self._start_websocket()
        
        # Also start polling as fallback
        poll_thread = threading.Thread(target=self._poll_blocks, daemon=True)
        poll_thread.start()
        
        try:
            # Keep running until stopped
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop subscribing and close connections."""
        print("\nStopping chain subscriber...")
        self.running = False
        
        # Unsubscribe if we have a subscription
        if self.ws and self.subscription_id:
            try:
                unsubscribe_msg = {
                    "jsonrpc": "2.0",
                    "method": "eth_unsubscribe",
                    "params": [self.subscription_id],
                    "id": 2
                }
                self.ws.send(json.dumps(unsubscribe_msg))
            except:
                pass
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
        
        # Close file
        if self.file_handle:
            footer = {
                "type": "footer",
                "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                "total_transactions_logged": self.transaction_count,
                "last_block_number": self.last_block_number
            }
            self.file_handle.write(json.dumps(footer, separators=(',', ':')) + "\n")
            self.file_handle.close()
            self.file_handle = None
        
        print(f"Logged {self.transaction_count} transactions")
        if self.log_file:
            print(f"Log saved to: {self.log_file}")
        
        # Display all collected EOA addresses
        print("\n" + "=" * 80)
        print("EOA (Externally Owned Account) ADDRESSES DETECTED")
        print("=" * 80)
        if self.eoa_addresses:
            print(f"Total unique EOA addresses: {len(self.eoa_addresses)}")
            print("\nEOA Addresses:")
            for i, address in enumerate(sorted(self.eoa_addresses), 1):
                print(f"  {i}. {address}")
        else:
            print("No EOA addresses detected.")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Subscribe to Hyperliquid on-chain block updates")
    parser.add_argument("--testnet", action="store_true", help="Use testnet endpoints")
    parser.add_argument("--log-file", type=str, help="File to log transactions to")
    parser.add_argument("--rpc-url", type=str, help="Custom RPC URL (HTTP or WebSocket)")
    args = parser.parse_args()
    
    subscriber = ChainSubscriber(testnet=args.testnet, log_file=args.log_file, rpc_url=args.rpc_url)
    
    def signal_handler(sig, frame):
        subscriber.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start subscribing
    subscriber.start()


if __name__ == "__main__":
    main()
