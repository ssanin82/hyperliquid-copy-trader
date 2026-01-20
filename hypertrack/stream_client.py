"""
StreamClient - WebSocket client for real-time Hyperliquid data streams.

This module provides a more efficient way to capture trading events
using WebSocket connections for lower latency and real-time updates.
"""

import json
import time
import websocket
import threading
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from queue import Queue


class HyperliquidStreamClient:
    """
    WebSocket client for Hyperliquid real-time data streams.
    
    Connects to Hyperliquid's WebSocket API to receive:
    - Real-time trades
    - Order book updates
    - User fills
    - Position updates
    
    This is more efficient than polling REST API endpoints.
    """
    
    def __init__(self, base_url: str = "wss://api.hyperliquid.xyz/ws"):
        """
        Initialize the WebSocket client.
        
        Args:
            base_url: WebSocket URL for Hyperliquid API
        """
        self.base_url = base_url
        self.ws = None
        self.running = False
        self.callbacks = {
            "trade": [],
            "fill": [],
            "l2_book": [],
            "position": []
        }
        self.message_queue = Queue()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
    
    def register_callback(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback for a specific event type.
        
        Args:
            event_type: Type of event (trade, fill, l2_book, position)
            callback: Function to call when event is received
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def _on_message(self, ws, message):
        """
        Handle incoming WebSocket messages.
        
        Args:
            ws: WebSocket connection
            message: Raw message string
        """
        try:
            data = json.loads(message)
            self.message_queue.put(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing message: {e}")
    
    def _on_error(self, ws, error):
        """
        Handle WebSocket errors.
        
        Args:
            ws: WebSocket connection
            error: Error object
        """
        print(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket close events.
        
        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
        """
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.running:
            self._reconnect()
    
    def _on_open(self, ws):
        """
        Handle WebSocket open events.
        
        Args:
            ws: WebSocket connection
        """
        print("WebSocket connected to Hyperliquid")
        self.reconnect_attempts = 0
        
        # Subscribe to trades
        # The exact subscription format depends on Hyperliquid's WebSocket API
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "trades"
            }
        }
        ws.send(json.dumps(subscribe_msg))
    
    def _reconnect(self):
        """
        Attempt to reconnect to the WebSocket.
        """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print("Max reconnection attempts reached. Stopping.")
            self.running = False
            return
        
        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff, max 60s
        print(f"Reconnecting in {wait_time} seconds... (attempt {self.reconnect_attempts})")
        time.sleep(wait_time)
        
        if self.running:
            self.connect()
    
    def _process_messages(self):
        """
        Process messages from the queue and trigger callbacks.
        """
        while self.running:
            try:
                data = self.message_queue.get(timeout=1)
                
                # Determine event type and trigger callbacks
                event_type = self._determine_event_type(data)
                
                if event_type and event_type in self.callbacks:
                    for callback in self.callbacks[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"Error in callback: {e}")
                            
            except Exception:
                continue
    
    def _determine_event_type(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Determine the event type from message data.
        
        Args:
            data: Message data dictionary
            
        Returns:
            Event type string or None
        """
        # This depends on Hyperliquid's WebSocket message format
        # Adjust based on actual API documentation
        if "channel" in data:
            channel = data["channel"]
            if channel == "trades":
                return "trade"
            elif channel == "fills":
                return "fill"
            elif channel == "l2Book":
                return "l2_book"
            elif channel == "position":
                return "position"
        
        return None
    
    def connect(self):
        """
        Connect to the WebSocket server.
        """
        try:
            self.ws = websocket.WebSocketApp(
                self.base_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Run WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            
        except Exception as e:
            print(f"Error connecting to WebSocket: {e}")
            if self.running:
                self._reconnect()
    
    def start(self):
        """
        Start the WebSocket client and message processing.
        """
        if self.running:
            return
        
        self.running = True
        
        # Start message processing thread
        process_thread = threading.Thread(target=self._process_messages, daemon=True)
        process_thread.start()
        
        # Connect to WebSocket
        self.connect()
    
    def stop(self):
        """
        Stop the WebSocket client.
        """
        self.running = False
        if self.ws:
            self.ws.close()
