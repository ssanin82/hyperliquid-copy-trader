"""
MetricsTracker - Tracks trading metrics per user wallet for profitable trader detection.

Subscribes to all necessary Hyperliquid streams and calculates:
- trades_per_day
- cancel_rate
- burstiness
- avg_leverage, max_leverage, leverage_volatility
- margin_delta_rate
- holding_time
- long_short_ratio
- flip_rate
"""

import json
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict, deque
import requests
import websocket
import threading
import statistics
import math

from hypertrack.constants import TOTAL_TRADES_THRESHOLD, WATCH_COINS


class UserMetrics:
    """Tracks metrics for a single user."""
    
    def __init__(self, user_address: str):
        self.user_address = user_address
        
        # Position tracking
        self.position_updates = []  # List of (timestamp, size_change, leverage, margin)
        self.open_positions = {}  # coin -> {open_time, size, side, leverage, margin}
        self.closed_positions = []  # List of {open_time, close_time, holding_time}
        
        # Order tracking
        self.limit_orders = 0
        self.cancel_orders = 0
        
        # Trade tracking
        self.trades = []  # List of (timestamp, side, size, coin)
        self.trade_times = []  # For burstiness calculation
        
        # Leverage tracking
        self.leverage_history = []  # List of leverage values
        
        # Margin tracking
        self.margin_history = []  # List of (timestamp, margin)
        self.margin_deltas = []  # List of absolute margin changes
        
        # Direction tracking (for flip_rate)
        self.last_direction = None  # "long" or "short"
        self.direction_changes = 0
        
        # Volume tracking
        self.long_volume = 0.0
        self.short_volume = 0.0
        
        # Coins tracked (for filtering by WATCH_COINS)
        self.coins_traded = set()
        
        # Start time for this user
        self.first_seen = time.time()
    
    def add_position_update(self, timestamp: float, coin: str, size_change: float, 
                          leverage: Optional[float] = None, margin: Optional[float] = None):
        """Record a position update."""
        if abs(size_change) > 0:  # Only count non-zero size changes
            self.coins_traded.add(coin)  # Track which coins this user trades
            self.position_updates.append({
                'timestamp': timestamp,
                'coin': coin,
                'size_change': size_change,
                'leverage': leverage,
                'margin': margin
            })
            self.trade_times.append(timestamp)
            
            # Track leverage
            if leverage is not None:
                self.leverage_history.append(leverage)
            
            # Track margin
            if margin is not None:
                if self.margin_history:
                    prev_margin = self.margin_history[-1][1]
                    delta = abs(margin - prev_margin)
                    if delta > 0:
                        self.margin_deltas.append(delta)
                self.margin_history.append((timestamp, margin))
            
            # Track position opens/closes
            if coin not in self.open_positions:
                # Opening new position
                if size_change != 0:
                    side = "long" if size_change > 0 else "short"
                    self.open_positions[coin] = {
                        'open_time': timestamp,
                        'size': abs(size_change),
                        'side': side,
                        'leverage': leverage or 1.0,
                        'margin': margin or 0.0
                    }
                    self.last_direction = side
            else:
                # Updating existing position
                pos = self.open_positions[coin]
                new_size = pos['size'] + size_change
                
                if abs(new_size) < 1e-8:  # Position closed
                    # Close position
                    close_time = timestamp
                    holding_time = close_time - pos['open_time']
                    self.closed_positions.append({
                        'open_time': pos['open_time'],
                        'close_time': close_time,
                        'holding_time': holding_time
                    })
                    del self.open_positions[coin]
                else:
                    # Update position
                    pos['size'] = abs(new_size)
                    if size_change < 0 and pos['side'] == 'long':
                        pos['side'] = 'short'
                        if self.last_direction == 'long':
                            self.direction_changes += 1
                        self.last_direction = 'short'
                    elif size_change > 0 and pos['side'] == 'short':
                        pos['side'] = 'long'
                        if self.last_direction == 'short':
                            self.direction_changes += 1
                        self.last_direction = 'long'
    
    def add_trade(self, timestamp: float, coin: str, side: str, size: float):
        """Record a trade."""
        self.coins_traded.add(coin)  # Track which coins this user trades
        self.trades.append({
            'timestamp': timestamp,
            'coin': coin,
            'side': side,
            'size': size
        })
        self.trade_times.append(timestamp)
        
        # Track volume
        if side.upper() in ['B', 'BUY', 'LONG']:
            self.long_volume += size
        else:
            self.short_volume += size
    
    def add_order_event(self, order_type: str):
        """Record an order event."""
        if order_type.lower() in ['limit', 'limitorder']:
            self.limit_orders += 1
        elif order_type.lower() in ['cancel', 'cancelorder']:
            self.cancel_orders += 1
    
    def calculate_metrics(self, session_duration_seconds: float) -> Dict[str, Any]:
        """Calculate all 10 metrics for this user."""
        metrics = {}
        
        # 1. trades_per_day = count(PositionUpdate where size change ≠ 0)
        # Already filtered in add_position_update
        trades_count = len(self.position_updates)
        days = session_duration_seconds / 86400.0  # Convert to days
        metrics['trades_per_day'] = trades_count / days if days > 0 else 0.0
        
        # 2. cancel_rate = CancelOrder / LimitOrder
        metrics['cancel_rate'] = (
            self.cancel_orders / self.limit_orders 
            if self.limit_orders > 0 else 0.0
        )
        
        # 3. burstiness = std(inter_event_time) / mean(inter_event_time)
        if len(self.trade_times) > 1:
            sorted_times = sorted(self.trade_times)
            inter_times = [
                sorted_times[i+1] - sorted_times[i] 
                for i in range(len(sorted_times) - 1)
            ]
            if inter_times:
                mean_inter = statistics.mean(inter_times)
                if mean_inter > 0:
                    std_inter = statistics.stdev(inter_times) if len(inter_times) > 1 else 0.0
                    metrics['burstiness'] = std_inter / mean_inter
                else:
                    metrics['burstiness'] = 0.0
            else:
                metrics['burstiness'] = 0.0
        else:
            metrics['burstiness'] = 0.0
        
        # 4. avg_leverage = mean(leverage)
        metrics['avg_leverage'] = (
            statistics.mean(self.leverage_history) 
            if self.leverage_history else 0.0
        )
        
        # 5. max_leverage = max(leverage)
        metrics['max_leverage'] = (
            max(self.leverage_history) 
            if self.leverage_history else 0.0
        )
        
        # 6. leverage_volatility = std(leverage)
        metrics['leverage_volatility'] = (
            statistics.stdev(self.leverage_history) 
            if len(self.leverage_history) > 1 else 0.0
        )
        
        # 7. margin_delta_rate = mean(|Δmargin|)
        metrics['margin_delta_rate'] = (
            statistics.mean(self.margin_deltas) 
            if self.margin_deltas else 0.0
        )
        
        # 8. holding_time = close_time - open_time (average)
        if self.closed_positions:
            avg_holding = statistics.mean([p['holding_time'] for p in self.closed_positions])
            metrics['avg_holding_time_seconds'] = avg_holding
            metrics['avg_holding_time_hours'] = avg_holding / 3600.0
        else:
            metrics['avg_holding_time_seconds'] = 0.0
            metrics['avg_holding_time_hours'] = 0.0
        
        # 9. long_short_ratio = long_volume / short_volume
        metrics['long_short_ratio'] = (
            self.long_volume / self.short_volume 
            if self.short_volume > 0 else (float('inf') if self.long_volume > 0 else 0.0)
        )
        
        # 10. flip_rate = direction_changes / trades
        total_trades = len(self.trades) + len(self.position_updates)
        metrics['flip_rate'] = (
            self.direction_changes / total_trades 
            if total_trades > 0 else 0.0
        )
        
        # Additional stats
        metrics['total_trades'] = total_trades
        metrics['total_position_updates'] = len(self.position_updates)
        metrics['total_closed_positions'] = len(self.closed_positions)
        metrics['open_positions'] = len(self.open_positions)
        metrics['limit_orders'] = self.limit_orders
        metrics['cancel_orders'] = self.cancel_orders
        metrics['long_volume'] = self.long_volume
        metrics['short_volume'] = self.short_volume
        metrics['direction_changes'] = self.direction_changes
        
        return metrics


class MetricsTracker:
    """
    Records all trading activity and calculates metrics per user.
    Subscribes to all necessary Hyperliquid streams to capture:
    - Trades
    - Orders (limit, market, cancel)
    - Position changes
    - Leverage updates
    - Margin changes
    - Withdrawals/deposits
    - All user events
    """
    
    MAINNET_API_URL = "https://api.hyperliquid.xyz"
    MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
    TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
    TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
    
    def __init__(self, log_file: str = "hyperliquid_activity.log", testnet: bool = False):
        """
        Initialize the metrics tracker.
        
        Args:
            log_file: File to record all activity to
            testnet: Use testnet endpoints
        """
        self.testnet = testnet
        self.api_url = self.TESTNET_API_URL if testnet else self.MAINNET_API_URL
        self.ws_url = self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL
        self.log_file = log_file
        self.log_file_handle = None
        
        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'hypertrack/0.1.0'
        })
        
        # WebSocket
        self.ws = None
        self.ws_thread = None
        self.running = False
        
        # User metrics tracking
        self.user_metrics: Dict[str, UserMetrics] = {}
        
        # Track discovered users
        self.tracked_users = set()
        
        # Session tracking
        self.start_time = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "trades": 0,
            "orders": 0,
            "position_updates": 0,
            "leverage_updates": 0,
            "margin_changes": 0,
            "other_events": 0
        }
    
    def _api_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request."""
        try:
            response = self.session.post(
                f"{self.api_url}/info",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error: {e}")
            return None
    
    def _get_user(self, user_address: str) -> UserMetrics:
        """Get or create UserMetrics for a user."""
        if user_address not in self.user_metrics:
            self.user_metrics[user_address] = UserMetrics(user_address)
        return self.user_metrics[user_address]
    
    def _record_event(self, event_type: str, data: Dict[str, Any]):
        """
        Record an event to the log file.
        
        Args:
            event_type: Type of event (trade, order, position_update, etc.)
            data: Event data
        """
        if self.log_file_handle is None:
            self.log_file_handle = open(self.log_file, "a", encoding="utf-8")
            # Write header
            header = {
                "type": "header",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "recorder_version": "0.1.0",
                "description": "Hyperliquid activity log - all events (trades, orders, positions, etc.)"
            }
            self.log_file_handle.write(json.dumps(header, separators=(',', ':')) + "\n")
            self.log_file_handle.flush()
        
        # Format event
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "unix_timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        
        # Write to log
        json_line = json.dumps(event, separators=(',', ':'))
        self.log_file_handle.write(json_line + "\n")
        self.log_file_handle.flush()
        
        self.stats["total_events"] += 1
        self.stats[event_type] = self.stats.get(event_type, 0) + 1
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if "channel" in data:
                channel = data["channel"]
                
                if channel == "trades" and "data" in data:
                    for trade in data["data"]:
                        self._handle_trade(trade)
                
                elif channel == "userEvents" and "data" in data:
                    # User events include position updates, order events, etc.
                    user = data.get("user")
                    if user:
                        for event in data["data"]:
                            self._handle_user_event(user, event)
                
                elif channel == "userFills" and "data" in data:
                    user = data.get("user")
                    if user:
                        for fill in data["data"]:
                            self._handle_user_fill(user, fill)
            
            # Also handle direct data (some messages might not have "channel")
            elif "data" in data and isinstance(data["data"], list):
                # Might be trades or other events
                for item in data["data"]:
                    if "coin" in item and "side" in item:
                        # Looks like a trade
                        self._handle_trade(item)
        
        except json.JSONDecodeError as e:
            print(f"Error parsing WebSocket message (not JSON): {e}")
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_trade(self, trade: Dict[str, Any]):
        """Handle trade event."""
        try:
            # Record the trade event
            self._record_event("trade", trade)
            self.stats["trades"] += 1
            
            # Extract users from trade
            users = trade.get("users", [])
            if not users and "user" in trade:
                users = [trade["user"]]
            
            coin = trade.get("coin", "")
            side = trade.get("side", "")
            size = float(trade.get("sz", 0))
            timestamp = trade.get("time", time.time() * 1000) / 1000.0  # Convert ms to seconds
            
            # Track trade for each user
            new_users_found = False
            for user in users:
                if user and user != "0x0000000000000000000000000000000000000000":
                    with self.lock:
                        if user not in self.user_metrics:
                            new_users_found = True
                        user_metrics = self._get_user(user)
                        user_metrics.add_trade(timestamp, coin, side, size)
                        self.tracked_users.add(user)
            
            # Print progress every 10 new users
            if new_users_found and len(self.user_metrics) % 10 == 0:
                print(f"Discovered {len(self.user_metrics)} users so far...")
        
        except Exception as e:
            print(f"Error handling trade: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_user_event(self, user: str, event: Dict[str, Any]):
        """Handle user event (position updates, orders, etc.)."""
        try:
            # Record the event
            event_with_user = {**event, "user": user}
            event_type_str = event.get("type", "user_event")
            self._record_event(event_type_str, event_with_user)
            
            with self.lock:
                user_metrics = self._get_user(user)
                self.tracked_users.add(user)
                
                event_type = event.get("type", "")
                timestamp = event.get("time", time.time() * 1000) / 1000.0
                
                # Position update
                if event_type in ["position", "positionUpdate", "positionUpdate"]:
                    self.stats["position_updates"] += 1
                    coin = event.get("coin", "")
                    size = float(event.get("size", 0))
                    prev_size = float(event.get("prevSize", 0))
                    size_change = size - prev_size
                    
                    leverage = event.get("leverage")
                    if leverage is not None:
                        leverage = float(leverage)
                        self.stats["leverage_updates"] += 1
                    
                    margin = event.get("margin")
                    if margin is not None:
                        margin = float(margin)
                        self.stats["margin_changes"] += 1
                    
                    user_metrics.add_position_update(
                        timestamp, coin, size_change, leverage, margin
                    )
                
                # Order event
                elif event_type in ["order", "orderUpdate", "limitOrder", "marketOrder", "cancelOrder"]:
                    self.stats["orders"] += 1
                    order_type = event.get("orderType", event_type)
                    user_metrics.add_order_event(order_type)
                
                # Withdrawal/Deposit
                elif event_type in ["withdrawal", "deposit", "transfer"]:
                    self.stats["other_events"] += 1
                    # Record but don't track in metrics (for now)
                
                else:
                    self.stats["other_events"] += 1
        
        except Exception as e:
            print(f"Error handling user event: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_user_fill(self, user: str, fill: Dict[str, Any]):
        """Handle user fill event."""
        try:
            # Record the fill
            fill_with_user = {**fill, "user": user}
            self._record_event("user_fill", fill_with_user)
            self.stats["trades"] += 1
            
            with self.lock:
                user_metrics = self._get_user(user)
                self.tracked_users.add(user)
                
                # Fills can also indicate position changes
                coin = fill.get("coin", "")
                size = float(fill.get("sz", 0))
                side = fill.get("side", "")
                timestamp = fill.get("time", time.time() * 1000) / 1000.0
                
                # Determine size change direction
                size_change = size if side.upper() in ['B', 'BUY', 'LONG'] else -size
                
                leverage = fill.get("leverage")
                margin = fill.get("margin")
                
                user_metrics.add_position_update(
                    timestamp, coin, size_change, leverage, margin
                )
                user_metrics.add_trade(timestamp, coin, side, size)
        
        except Exception as e:
            print(f"Error handling user fill: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_ws_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.running:
            time.sleep(2)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        print("WebSocket connected to Hyperliquid")
        
        # Use WATCH_COINS from constants
        coins = WATCH_COINS
        
        # Subscribe to trades for watched coins (to discover users)
        print(f"Subscribing to trades for {len(coins)} coins: {', '.join(coins)}")
        subscribed = 0
        for coin in coins:
            try:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": coin}
                }
                ws.send(json.dumps(subscribe_msg))
                subscribed += 1
                if subscribed % 10 == 0:
                    print(f"  Subscribed to {subscribed}/{len(coins)} coins...")
                time.sleep(0.05)  # Small delay
            except Exception as e:
                print(f"Error subscribing to {coin}: {e}")
        
        print(f"Subscribed to {subscribed} trade streams. Discovering users...")
        print("Waiting for trades to discover users...")
    
    def _subscribe_to_user_events(self, ws, user: str):
        """Subscribe to user-specific events."""
        try:
            # Subscribe to user events (position updates, orders, etc.)
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "userEvents", "user": user}
            }
            ws.send(json.dumps(subscribe_msg))
        except Exception as e:
            print(f"Error subscribing to user events for {user[:10]}...: {e}")
    
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
    
    def _subscribe_to_new_users(self):
        """Periodically subscribe to events for newly discovered users."""
        while self.running:
            try:
                time.sleep(5)  # Check every 5 seconds
                
                # Check if WebSocket is connected
                ws_connected = False
                try:
                    if self.ws and hasattr(self.ws, 'sock') and self.ws.sock:
                        ws_connected = self.ws.sock.connected
                except:
                    pass
                
                if ws_connected:
                    with self.lock:
                        # Get users that have been discovered but not yet subscribed
                        subscribed_users = set(self.user_metrics.keys())
                        new_users = self.tracked_users - subscribed_users
                        
                        if new_users:
                            users_to_subscribe = list(new_users)[:10]  # Limit to 10 at a time
                            print(f"Subscribing to events for {len(users_to_subscribe)} new users...")
                            for user in users_to_subscribe:
                                self._subscribe_to_user_events(self.ws, user)
                                time.sleep(0.2)
                else:
                    # WebSocket not connected, wait longer
                    time.sleep(10)
            
            except Exception as e:
                print(f"Error in user subscription loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    def start(self):
        """Start tracking metrics."""
        if self.running:
            print("Tracker is already running!")
            return
        
        print("=" * 80)
        print("Hyperliquid Metrics Tracker")
        print("=" * 80)
        print(f"Recording all activity to: {self.log_file}")
        print("Tracking metrics for all discovered users...")
        print("Press Ctrl+C to stop and generate report")
        print("=" * 80)
        
        self.running = True
        self.start_time = time.time()
        
        # Start WebSocket
        self._start_websocket()
        
        # Start user subscription thread
        user_sub_thread = threading.Thread(target=self._subscribe_to_new_users, daemon=True)
        user_sub_thread.start()
        
        try:
            # Keep running until stopped
            last_status_time = time.time()
            while self.running:
                time.sleep(1)
                
                # Print status every 30 seconds
                if time.time() - last_status_time > 30:
                    with self.lock:
                        # Count eligible users (those with more than TOTAL_TRADES_THRESHOLD trades AND activity in WATCH_COINS)
                        eligible_count = 0
                        for metrics_obj in self.user_metrics.values():
                            # Check if user has activity in WATCH_COINS
                            has_watched_coin = bool(metrics_obj.coins_traded & set(WATCH_COINS))
                            if not has_watched_coin:
                                continue
                            
                            total_trades = len(metrics_obj.trades) + len(metrics_obj.position_updates)
                            if total_trades > TOTAL_TRADES_THRESHOLD:
                                eligible_count += 1
                        
                        total_trades = sum(len(m.trades) for m in self.user_metrics.values())
                        print(f"\n[Status] Events recorded: {self.stats['total_events']}, "
                              f"Users discovered: {len(self.tracked_users)}, "
                              f"Users with metrics: {len(self.user_metrics)}, "
                              f"Eligible users count: {eligible_count}, "
                              f"Trades: {self.stats['trades']}, "
                              f"Orders: {self.stats['orders']}, "
                              f"Position updates: {self.stats['position_updates']}")
                    last_status_time = time.time()
        except KeyboardInterrupt:
            print("\n\nStopping tracker...")
            self.stop()
    
    def stop(self):
        """Stop tracking and generate report."""
        if not self.running:
            return
        
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        # Close log file
        if self.log_file_handle:
            # Write footer
            footer = {
                "type": "footer",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "statistics": self.stats,
                "total_users": len(self.user_metrics)
            }
            self.log_file_handle.write(json.dumps(footer, separators=(',', ':')) + "\n")
            self.log_file_handle.close()
            self.log_file_handle = None
        
        # Calculate session duration
        session_duration = time.time() - self.start_time if self.start_time else 0
        
        # Generate report
        self._generate_report(session_duration)
        
        self.session.close()
    
    def _generate_report(self, session_duration: float):
        """Generate metrics report for all users."""
        print("\n" + "=" * 80)
        print("METRICS REPORT")
        print("=" * 80)
        print(f"Session duration: {session_duration / 3600:.2f} hours ({session_duration:.0f} seconds)")
        print(f"Total events recorded: {self.stats['total_events']}")
        print(f"  - Trades: {self.stats['trades']}")
        print(f"  - Orders: {self.stats['orders']}")
        print(f"  - Position updates: {self.stats['position_updates']}")
        print(f"  - Leverage updates: {self.stats['leverage_updates']}")
        print(f"  - Margin changes: {self.stats['margin_changes']}")
        print(f"  - Other events: {self.stats['other_events']}")
        print(f"Total users tracked: {len(self.user_metrics)}")
        print(f"Total users discovered: {len(self.tracked_users)}")
        print(f"Activity log saved to: {self.log_file}")
        print("=" * 80)
        
        if not self.user_metrics:
            print("No users tracked.")
            print("\nPossible reasons:")
            print("- No trades received (WebSocket may not be connected)")
            print("- Users discovered but no events received")
            print("- Check WebSocket connection status")
            
            # Still save empty report
            report_file = f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump({
                    'session_duration_seconds': session_duration,
                    'total_users': 0,
                    'total_users_discovered': len(self.tracked_users),
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                    'users': []
                }, f, indent=2)
            print(f"\nEmpty report saved to: {report_file}")
            return
        
        # Calculate metrics for all users
        user_reports = []
        for user, metrics_obj in self.user_metrics.items():
            # Only include users with activity in WATCH_COINS
            has_watched_coin = bool(metrics_obj.coins_traded & set(WATCH_COINS))
            if not has_watched_coin:
                continue
            
            metrics = metrics_obj.calculate_metrics(session_duration)
            metrics['user_address'] = user
            user_reports.append(metrics)
        
        # Filter: only show users with more than TOTAL_TRADES_THRESHOLD trades
        filtered_reports = [r for r in user_reports if r['total_trades'] > TOTAL_TRADES_THRESHOLD]
        
        if not filtered_reports:
            print(f"\nNo users with more than {TOTAL_TRADES_THRESHOLD} trades.")
            print(f"Total users tracked: {len(user_reports)}")
            print(f"Users filtered out (≤{TOTAL_TRADES_THRESHOLD} trades): {len(user_reports) - len(filtered_reports)}")
            
            # Still save report with all users (but note the filter)
            report_file = f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump({
                    'session_duration_seconds': session_duration,
                    'total_users': len(user_reports),
                    'total_users_discovered': len(self.tracked_users),
                    'users_shown': len(filtered_reports),
                    'filter_applied': f'users_with_more_than_{TOTAL_TRADES_THRESHOLD}_trades',
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                    'users': filtered_reports
                }, f, indent=2)
            print(f"\nReport saved to: {report_file} (filtered, showing only users with >{TOTAL_TRADES_THRESHOLD} trades)")
            return
        
        # Sort by total trades (most active first)
        filtered_reports.sort(key=lambda x: x['total_trades'], reverse=True)
        
        print(f"\nShowing {len(filtered_reports)} users with more than {TOTAL_TRADES_THRESHOLD} trades in WATCH_COINS")
        print(f"(Filtered out {len(user_reports) - len(filtered_reports)} users with ≤{TOTAL_TRADES_THRESHOLD} trades)")
        print(f"WATCH_COINS: {', '.join(WATCH_COINS)}")
        print("=" * 80)
        
        # Print report for each user
        for i, report in enumerate(filtered_reports, 1):
            user = report['user_address']
            print(f"\n{'=' * 80}")
            print(f"User #{i}: {user}")
            print(f"{'=' * 80}")
            print(f"1.  trades_per_day:           {report['trades_per_day']:.2f}")
            print(f"2.  cancel_rate:              {report['cancel_rate']:.4f}")
            print(f"3.  burstiness:                {report['burstiness']:.4f}")
            print(f"4.  avg_leverage:             {report['avg_leverage']:.2f}x")
            print(f"5.  max_leverage:             {report['max_leverage']:.2f}x")
            print(f"6.  leverage_volatility:      {report['leverage_volatility']:.4f}")
            print(f"7.  margin_delta_rate:        ${report['margin_delta_rate']:.2f}")
            print(f"8.  avg_holding_time:          {report['avg_holding_time_hours']:.2f} hours")
            print(f"9.  long_short_ratio:         {report['long_short_ratio']:.4f}")
            print(f"10. flip_rate:                {report['flip_rate']:.4f}")
            print(f"\nAdditional Stats:")
            print(f"    Total trades:             {report['total_trades']}")
            print(f"    Position updates:         {report['total_position_updates']}")
            print(f"    Closed positions:         {report['total_closed_positions']}")
            print(f"    Open positions:           {report['open_positions']}")
            print(f"    Limit orders:             {report['limit_orders']}")
            print(f"    Cancel orders:            {report['cancel_orders']}")
            print(f"    Long volume:              {report['long_volume']:.2f}")
            print(f"    Short volume:             {report['short_volume']:.2f}")
            print(f"    Direction changes:        {report['direction_changes']}")
        
        # Save report to file (only users with >TOTAL_TRADES_THRESHOLD trades)
        report_file = f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'session_duration_seconds': session_duration,
                'total_users_tracked': len(self.user_metrics),
                'total_users_discovered': len(self.tracked_users),
                    'users_shown': len(filtered_reports),
                    'users_filtered_out': len(user_reports) - len(filtered_reports),
                    'filter_applied': f'users_with_more_than_{TOTAL_TRADES_THRESHOLD}_trades_in_WATCH_COINS',
                    'trades_threshold': TOTAL_TRADES_THRESHOLD,
                    'watch_coins': WATCH_COINS,
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                    'users': filtered_reports
            }, f, indent=2)
        
        # Print summary
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total users shown: {len(filtered_reports)}")
        print(f"{'=' * 80}")
        print(f"Report saved to: {report_file}")
        print(f"{'=' * 80}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Record all trading activity and calculate metrics per user wallet"
    )
    parser.add_argument(
        "-f", "--log-file",
        default="hyperliquid_activity.log",
        help="File to record all activity to (default: hyperliquid_activity.log)"
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet endpoints"
    )
    
    args = parser.parse_args()
    
    tracker = MetricsTracker(log_file=args.log_file, testnet=args.testnet)
    
    def signal_handler(sig, frame):
        tracker.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        tracker.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        tracker.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
