"""
WalletProfiler - ML-based wallet profiling and classification system.

Uses PyTorch multi-task learning to derive:
- Trading style score
- Risk score
- Profitability score
- Bot probability
- Influence score
- Sophistication score

Subscribes to blockchain events (NOT userEvents/userTrades - those are private).
"""

import json
import time
import signal
import sys
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
import requests
import websocket
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import statistics
import math


NO_PUBLIC_API = False
API_TOKENS = [
    "HYPE",      # Hyperliquid native token
    "BTC",       # Bitcoin
    "ETH",       # Ethereum
    "SOL",       # Solana
    "AVAX",      # Avalanche
    "OP",        # Optimism
    "BNB",       # Binance Coin
    "DOGE",      # Dogecoin
    "LINK",      # Chainlink
    "MATIC",     # Polygon
    "LTC",       # Litecoin
    "ARB",       # Arbitrum
    "FTM",       # Fantom
    "XRP",       # Ripple
    "ETC",       # Ethereum Classic
    "APE",       # ApeCoin
    "PUMP",      # Pump token (high volume meme token) 
    "PURR",      # Hypurr Fun / PURR
    "KNTQ",      # Kinetiq
    "SEDA"       # SEDA Protocol token
]


class WalletFeatures:
    """Tracks features for a single wallet."""
    
    def __init__(self, address: str):
        self.address = address.lower()
        self.first_seen = time.time()
        self.last_seen = time.time()
        
        # Transaction features
        self.transactions = []
        self.transaction_times = []
        self.transaction_values = []
        
        # ERC-20 transfer features
        self.erc20_transfers = []
        self.tokens_traded = set()
        self.total_volume = 0.0
        
        # Temporal features
        self.active_hours = set()
        self.session_starts = []
        
        # Directional features
        self.long_count = 0
        self.short_count = 0
        self.direction_changes = 0
        self.last_direction = None
        
        # Size features
        self.trade_sizes = []
        
        # Risk features
        self.leverage_used = []
        self.margin_changes = []
        
        # Network features
        self.interacted_addresses = set()
        self.contract_interactions = set()
        
        # Market data features (from Hyperliquid API)
        self.trades_with_market_data = []  # List of (trade, market_state) tuples
        self.market_impact_scores = []  # Market impact per trade
        self.slippage_scores = []  # Slippage per trade
        self.momentum_scores = []  # Momentum following score
        self.mean_reversion_scores = []  # Mean reversion score
        self.volatility_scores = []  # Volatility trading score
        self.market_maker_scores = []  # Market making activity
        self.order_book_participation = []  # Order book participation
        
    def add_transaction(self, tx_data: Dict[str, Any], timestamp: float):
        """Add a transaction to wallet history."""
        self.transactions.append(tx_data)
        self.transaction_times.append(timestamp)
        self.last_seen = timestamp
        
        # Extract value if available
        value = tx_data.get("value", "0x0")
        if isinstance(value, str):
            try:
                value_wei = int(value, 16)
                value_eth = value_wei / 1e18
                self.transaction_values.append(value_eth)
            except:
                pass
        
        # Track active hours
        dt = datetime.fromtimestamp(timestamp, UTC)
        self.active_hours.add(dt.hour)
        
        # Track interactions
        tx_from = (tx_data.get("from") or "").lower()
        tx_to = (tx_data.get("to") or "").lower()
        if tx_from == self.address and tx_to:
            self.interacted_addresses.add(tx_to)
        elif tx_to == self.address and tx_from:
            self.interacted_addresses.add(tx_from)
    
    def add_erc20_transfer(self, transfer_data: Dict[str, Any], timestamp: float):
        """Add an ERC-20 transfer."""
        self.erc20_transfers.append(transfer_data)
        token_address = (transfer_data.get("address") or "").lower()
        if token_address:
            self.tokens_traded.add(token_address)
        
        # Extract value
        value_data = transfer_data.get("value", "0x0")
        if isinstance(value_data, str):
            try:
                value = int(value_data, 16)
                self.total_volume += value
            except:
                pass
    
    def extract_features(self) -> Dict[str, float]:
        """Extract ML features from wallet history."""
        features = {}
        
        # Basic counts
        features['tx_count'] = len(self.transactions)
        features['erc20_count'] = len(self.erc20_transfers)
        features['unique_tokens'] = len(self.tokens_traded)
        features['unique_addresses'] = len(self.interacted_addresses)
        
        # Temporal features
        if self.transaction_times:
            features['age_days'] = (self.last_seen - self.first_seen) / 86400
            features['tx_per_day'] = features['tx_count'] / max(features['age_days'], 0.01)
            
            # Burstiness
            if len(self.transaction_times) > 1:
                sorted_times = sorted(self.transaction_times)
                inter_times = [sorted_times[i+1] - sorted_times[i] 
                              for i in range(len(sorted_times) - 1)]
                if inter_times:
                    mean_inter = statistics.mean(inter_times)
                    if mean_inter > 0:
                        std_inter = statistics.stdev(inter_times) if len(inter_times) > 1 else 0.0
                        features['burstiness'] = std_inter / mean_inter
                    else:
                        features['burstiness'] = 0.0
                else:
                    features['burstiness'] = 0.0
            else:
                features['burstiness'] = 0.0
            
            # Active hours entropy
            if self.transaction_times:
                # Count transactions per hour
                hour_counts = [0] * 24
                for tx_time in self.transaction_times:
                    dt = datetime.fromtimestamp(tx_time, UTC)
                    hour_counts[dt.hour] += 1
                total = sum(hour_counts)
                if total > 0:
                    entropy = -sum((c/total) * math.log2(c/total) 
                                  for c in hour_counts if c > 0)
                    features['hour_entropy'] = entropy / math.log2(24) if math.log2(24) > 0 else 0.0
                else:
                    features['hour_entropy'] = 0.0
            else:
                features['hour_entropy'] = 0.0
        else:
            features['age_days'] = 0.0
            features['tx_per_day'] = 0.0
            features['burstiness'] = 0.0
            features['hour_entropy'] = 0.0
        
        # Value features
        if self.transaction_values:
            features['total_value'] = sum(self.transaction_values)
            features['avg_value'] = statistics.mean(self.transaction_values)
            features['max_value'] = max(self.transaction_values)
            features['value_std'] = statistics.stdev(self.transaction_values) if len(self.transaction_values) > 1 else 0.0
        else:
            features['total_value'] = 0.0
            features['avg_value'] = 0.0
            features['max_value'] = 0.0
            features['value_std'] = 0.0
        
        # ERC-20 features
        features['erc20_volume'] = self.total_volume
        features['token_diversity'] = len(self.tokens_traded)
        
        # Directional features
        total_directional = self.long_count + self.short_count
        if total_directional > 0:
            features['long_ratio'] = self.long_count / total_directional
            features['flip_rate'] = self.direction_changes / total_directional
        else:
            features['long_ratio'] = 0.5
            features['flip_rate'] = 0.0
        
        # Size entropy (for bot detection)
        if self.trade_sizes:
            if len(self.trade_sizes) > 1:
                min_size = min(self.trade_sizes)
                max_size = max(self.trade_sizes)
                if max_size > min_size:
                    bins = 10
                    bin_counts = [0] * bins
                    bin_size = (max_size - min_size) / bins
                    for s in self.trade_sizes:
                        bin_idx = min(int((s - min_size) / bin_size), bins - 1)
                        bin_counts[bin_idx] += 1
                    total = len(self.trade_sizes)
                    entropy = -sum((c/total) * math.log2(c/total) 
                                  for c in bin_counts if c > 0)
                    features['size_entropy'] = entropy / math.log2(bins) if math.log2(bins) > 0 else 0.0
                else:
                    features['size_entropy'] = 0.0
            else:
                features['size_entropy'] = 0.0
        else:
            features['size_entropy'] = 1.0
        
        # Network features
        features['contract_interaction_count'] = len(self.contract_interactions)
        features['address_diversity'] = len(self.interacted_addresses)
        
        # Market data features
        if self.market_impact_scores:
            features['avg_market_impact'] = statistics.mean(self.market_impact_scores)
            features['max_market_impact'] = max(self.market_impact_scores)
        else:
            features['avg_market_impact'] = 0.0
            features['max_market_impact'] = 0.0
        
        if self.slippage_scores:
            features['avg_slippage'] = statistics.mean(self.slippage_scores)
            features['slippage_std'] = statistics.stdev(self.slippage_scores) if len(self.slippage_scores) > 1 else 0.0
        else:
            features['avg_slippage'] = 0.0
            features['slippage_std'] = 0.0
        
        if self.momentum_scores:
            features['momentum_score'] = statistics.mean(self.momentum_scores)
        else:
            features['momentum_score'] = 0.0
        
        if self.mean_reversion_scores:
            features['mean_reversion_score'] = statistics.mean(self.mean_reversion_scores)
        else:
            features['mean_reversion_score'] = 0.0
        
        if self.volatility_scores:
            features['volatility_trading_score'] = statistics.mean(self.volatility_scores)
        else:
            features['volatility_trading_score'] = 0.0
        
        if self.market_maker_scores:
            features['market_maker_score'] = statistics.mean(self.market_maker_scores)
        else:
            features['market_maker_score'] = 0.0
        
        if self.order_book_participation:
            features['order_book_participation'] = statistics.mean(self.order_book_participation)
        else:
            features['order_book_participation'] = 0.0
        
        return features
    
    def classify_wallet(self, features: Dict[str, float], bot_probability: float = 0.0) -> List[tuple]:
        """
        Classify wallet into behavior categories.
        
        Returns list of (category, confidence) tuples.
        """
        categories = []
        
        tx_per_day = features.get('tx_per_day', 0.0)
        size_entropy = features.get('size_entropy', 1.0)
        burstiness = features.get('burstiness', 0.0)
        age_days = features.get('age_days', 0.0)
        tx_count = features.get('tx_count', 0.0)
        max_value = features.get('max_value', 0.0)
        total_value = features.get('total_value', 0.0)
        avg_value = features.get('avg_value', 0.0)
        unique_tokens = features.get('unique_tokens', 0.0)
        erc20_count = features.get('erc20_count', 0.0)
        flip_rate = features.get('flip_rate', 0.0)
        
        # Bot classification
        if bot_probability > 0.7:
            categories.append(("Bot", bot_probability))
        elif bot_probability > 0.5:
            categories.append(("Possible Bot", bot_probability))
        
        # Scalper: High frequency, small sizes, high burstiness
        if tx_per_day > 50 and size_entropy < 0.4:
            confidence = min(1.0, (tx_per_day / 100.0) * (1.0 - size_entropy))
            categories.append(("Scalper", confidence))
        elif tx_per_day > 30 and size_entropy < 0.5:
            categories.append(("Possible Scalper", 0.6))
        
        # HODLer: Very low frequency, old wallet
        if tx_per_day < 0.1 and age_days > 30:
            confidence = min(1.0, (30.0 / max(age_days, 1.0)) * (0.1 / max(tx_per_day, 0.01)))
            categories.append(("HODLer", confidence))
        elif tx_per_day < 0.5 and age_days > 60:
            categories.append(("Possible HODLer", 0.6))
        
        # Whale: Large transaction values
        # Calculate percentile thresholds (simplified - in production use actual percentiles)
        if max_value > 100.0 or total_value > 1000.0:  # Adjust thresholds based on actual data
            confidence = min(1.0, (max_value / 1000.0) if max_value > 0 else (total_value / 10000.0))
            categories.append(("Whale", min(1.0, confidence)))
        elif max_value > 10.0 or total_value > 100.0:
            categories.append(("Possible Whale", 0.6))
        
        # Token Collector: High ERC-20 activity, many tokens
        if unique_tokens > 10 and erc20_count > tx_count * 2:
            confidence = min(1.0, (unique_tokens / 50.0) * (erc20_count / max(tx_count * 3, 1.0)))
            categories.append(("Token Collector", confidence))
        elif unique_tokens > 5 and erc20_count > tx_count:
            categories.append(("Possible Token Collector", 0.6))
        
        # Active Trader (could include Swing Traders): Moderate-high frequency
        if 1.0 <= tx_per_day <= 50 and burstiness > 0.3:
            confidence = min(1.0, tx_per_day / 50.0)
            categories.append(("Active Trader", confidence))
        elif 0.5 <= tx_per_day <= 30:
            categories.append(("Moderate Trader", 0.6))
        
        # Arbitrageur: Multiple tokens, high flip rate
        if unique_tokens > 5 and flip_rate > 0.5:
            confidence = min(1.0, (unique_tokens / 20.0) * flip_rate)
            categories.append(("Arbitrageur", confidence))
        elif unique_tokens > 3 and flip_rate > 0.3:
            categories.append(("Possible Arbitrageur", 0.6))
        
        # Dormant: Very low activity, old wallet
        if tx_count < 5 and age_days > 30:
            confidence = min(1.0, (30.0 / max(age_days, 1.0)) * ((5.0 - tx_count) / 5.0))
            categories.append(("Dormant", confidence))
        
        # New categories from market data
        avg_market_impact = features.get('avg_market_impact', 1.0)
        max_market_impact = features.get('max_market_impact', 1.0)
        avg_slippage = features.get('avg_slippage', 0.0)
        momentum_score = features.get('momentum_score', 0.0)
        mean_reversion_score = features.get('mean_reversion_score', 0.0)
        volatility_score = features.get('volatility_trading_score', 0.0)
        market_maker_score = features.get('market_maker_score', 0.0)
        order_book_participation = features.get('order_book_participation', 0.0)
        
        # Stealth Trader: Low impact, intentional stealth
        if avg_market_impact < 0.1 and max_market_impact < 0.2 and tx_count > 10:
            confidence = min(1.0, (0.1 - avg_market_impact) * 10)
            categories.append(("Stealth Trader", confidence))
        elif avg_market_impact < 0.2 and max_market_impact < 0.3:
            categories.append(("Possible Stealth Trader", 0.6))
        
        # Smart Router: Large size with minimal impact
        if avg_value > 10.0 and avg_market_impact < 0.15 and tx_count > 5:
            confidence = min(1.0, (avg_value / 100.0) * (0.15 - avg_market_impact) * 10)
            categories.append(("Smart Router", confidence))
        elif avg_value > 5.0 and avg_market_impact < 0.2:
            categories.append(("Possible Smart Router", 0.6))
        
        # Volatility Arbitrageur: Exploits volatility mispricings
        if volatility_score > 0.7 and mean_reversion_score > 0.6 and tx_count > 10:
            confidence = min(1.0, (volatility_score + mean_reversion_score) / 2)
            categories.append(("Volatility Arbitrageur", confidence))
        elif volatility_score > 0.6 and mean_reversion_score > 0.5:
            categories.append(("Possible Volatility Arbitrageur", 0.6))
        
        # Hybrid Strategist: Adapts between momentum and mean reversion
        momentum_strength = abs(momentum_score - 0.5)
        mean_rev_strength = abs(mean_reversion_score - 0.5)
        if momentum_strength < 0.3 and mean_rev_strength < 0.3 and tx_count > 15:
            # Uses both strategies, adapts
            confidence = min(1.0, 1.0 - (momentum_strength + mean_rev_strength))
            categories.append(("Hybrid Strategist", confidence))
        elif momentum_strength < 0.4 and mean_rev_strength < 0.4:
            categories.append(("Possible Hybrid Strategist", 0.6))
        
        # Sophisticated Market Maker: Advanced market making with impact management
        if market_maker_score > 0.7 and order_book_participation > 0.6 and avg_market_impact < 0.2:
            confidence = min(1.0, (market_maker_score + order_book_participation) / 2 * (1.0 - avg_market_impact * 2))
            categories.append(("Sophisticated Market Maker", confidence))
        elif market_maker_score > 0.6 and order_book_participation > 0.5:
            categories.append(("Possible Sophisticated Market Maker", 0.6))
        
        # Sort by confidence (highest first)
        categories.sort(key=lambda x: x[1], reverse=True)
        
        return categories


class WalletMTLModel(nn.Module):
    """Multi-Task Learning model for wallet profiling."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 256):
        super(WalletMTLModel, self).__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Task-specific heads
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 6D style vector
            nn.Tanh()  # Bounded output
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 risk score
        )
        
        self.profitability_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 profitability score
        )
        
        self.bot_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 bot probability
        )
        
        self.influence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Non-negative influence score
        )
        
        self.sophistication_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 sophistication score
        )
    
    def forward(self, x):
        shared = self.encoder(x)
        return {
            'style': self.style_head(shared),
            'risk': self.risk_head(shared),
            'profitability': self.profitability_head(shared),
            'bot': self.bot_head(shared),
            'influence': self.influence_head(shared),
            'sophistication': self.sophistication_head(shared)
        }


class WalletDataset(Dataset):
    """Dataset for wallet features."""
    
    def __init__(self, features_list: List[np.ndarray], labels: Optional[Dict[str, List]] = None):
        self.features = torch.FloatTensor(features_list)
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {'features': self.features[idx]}
        if self.labels:
            item.update({k: torch.FloatTensor(v[idx]) for k, v in self.labels.items()})
        return item


class WalletProfiler:
    """Profiles wallets using ML and blockchain data."""
    
    MAINNET_RPC_HTTP = "https://hyperliquid-mainnet.g.alchemy.com/v2/E45W-MsgmrM0Ye2gH8ZoX"
    MAINNET_RPC_WS = "wss://hyperliquid-mainnet.g.alchemy.com/v2/E45W-MsgmrM0Ye2gH8ZoX"
    # Hyperliquid API WebSocket for market data
    HYPERLIQUID_API_WS = "wss://api.hyperliquid.xyz/ws"
    HYPERLIQUID_API_HTTP = "https://api.hyperliquid.xyz"
    
    def __init__(self, log_file: str = "wallet_profiling.log", model_file: str = "wallet_model.pt"):
        self.log_file = log_file
        self.model_file = model_file
        self.file_handle = None
        
        # Wallet tracking
        self.wallets: Dict[str, WalletFeatures] = {}
        self.eoa_addresses = set()
        self.erc20_tokens = set()
        
        # Model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Strategy start time for periodic logging
        self.strategy_start_time = None
        self.status_log_thread = None
        self.feature_names = [
            'tx_count', 'erc20_count', 'unique_tokens', 'unique_addresses',
            'age_days', 'tx_per_day', 'burstiness', 'hour_entropy',
            'total_value', 'avg_value', 'max_value', 'value_std',
            'erc20_volume', 'token_diversity', 'long_ratio', 'flip_rate',
            'size_entropy', 'contract_interaction_count', 'address_diversity',
            'avg_market_impact', 'max_market_impact', 'avg_slippage', 'slippage_std',
            'momentum_score', 'mean_reversion_score', 'volatility_trading_score',
            'market_maker_score', 'order_book_participation'
        ]
        
        # Training data
        self.training_data = []
        self.running = False
        self.rpc_call_count = 0
        self.rpc_start_time = None
        self.subscription_id = None
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # WebSocket (blockchain RPC)
        self.ws = None
        self.ws_thread = None
        
        # Hyperliquid API WebSocket for market data
        self.api_ws = None
        self.api_ws_thread = None
        
        # Market data storage
        self.market_data = {
            'l2_book': {},  # coin -> latest order book
            'bbo': {},  # coin -> best bid/offer
            'candle': {},  # coin -> latest candle
            'all_mids': {},  # coin -> mid price
            'trades': []  # recent trades
        }
        self.market_data_lock = threading.Lock()
        self.available_coins = []  # List of all available coins
        self.coins_subscribed = set()  # Track which coins we've subscribed to
        self.api_connection_count = 0  # Track number of API WebSocket connections
        self.max_api_connections = 100  # Maximum allowed connections
        self.total_subscriptions_made = 0  # Track total number of subscriptions
        
        # Statistics for final report
        self.api_message_stats = {}  # {channel: {coin: count}} - messages per channel per token
        self.wallets_in_trades = set()  # Wallets detected on-chain that were involved in API trades
    
    def _format_elapsed_time(self, elapsed_seconds: float) -> str:
        """Format elapsed time as 'X hours, Y minutes, Z seconds passed'."""
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or len(parts) == 0:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        
        return ", ".join(parts) + " passed"
    
    def _periodic_status_log(self):
        """Log elapsed time every minute."""
        while self.running:
            time.sleep(60)  # Wait 60 seconds
            if self.running and self.strategy_start_time:
                elapsed = time.time() - self.strategy_start_time
                elapsed_str = self._format_elapsed_time(elapsed)
                wallet_count = len(self.wallets)
                print(f"\n[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC] {elapsed_str} | Wallets detected: {wallet_count}")
                if self.file_handle:
                    try:
                        status_log = {
                            "type": "status",
                            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                            "elapsed_time": elapsed,
                            "elapsed_time_formatted": elapsed_str,
                            "wallet_count": wallet_count
                        }
                        self.file_handle.write(json.dumps(status_log, separators=(',', ':')) + "\n")
                        self.file_handle.flush()
                    except (ValueError, OSError):
                        pass
        
    def _log(self, message: str, data: Optional[Dict] = None):
        """Log message and data."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "message": message
        }
        if data:
            log_entry["data"] = data
        
        print(f"[{log_entry['timestamp']}] {message}")
        if self.file_handle:
            try:
                self.file_handle.write(json.dumps(log_entry, separators=(',', ':')) + "\n")
                self.file_handle.flush()
            except (ValueError, OSError):
                # File is closed or error writing, ignore
                pass
    
    def _json_rpc_request(self, method: str, params: list) -> Optional[Dict[str, Any]]:
        """Make JSON-RPC request."""
        if self.rpc_start_time is None:
            self.rpc_start_time = time.time()
        self.rpc_call_count += 1
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        try:
            response = requests.post(
                self.MAINNET_RPC_HTTP,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    return None
                return result.get("result")
        except Exception as e:
            self._log(f"RPC error: {e}")
        return None
    
    def _is_eoa(self, address: str) -> bool:
        """Check if address is EOA."""
        if not address or address == "0x" or address == "0x0":
            return False
        address = address.lower()
        
        code = self._json_rpc_request("eth_getCode", [address, "latest"])
        return code is None or code == "0x" or code == "" or code == "0x0"
    
    def _process_transaction(self, tx: Dict[str, Any], block_timestamp: float):
        """Process a transaction and update wallet features."""
        tx_from = (tx.get("from") or "").lower()
        tx_to = (tx.get("to") or "").lower()
        
        # Process from address
        if tx_from and tx_from != "0x0":
            if self._is_eoa(tx_from):
                self.eoa_addresses.add(tx_from)
                if tx_from not in self.wallets:
                    self.wallets[tx_from] = WalletFeatures(tx_from)
                self.wallets[tx_from].add_transaction(tx, block_timestamp)
                # Try to correlate with market data
                self._correlate_trade_with_market_data(tx_from, tx, block_timestamp)
            else:
                # Contract interaction
                if tx_from in self.wallets:
                    self.wallets[tx_from].contract_interactions.add(tx_from)
        
        # Process to address
        if tx_to and tx_to != "0x0":
            if self._is_eoa(tx_to):
                self.eoa_addresses.add(tx_to)
                if tx_to not in self.wallets:
                    self.wallets[tx_to] = WalletFeatures(tx_to)
                self.wallets[tx_to].add_transaction(tx, block_timestamp)
                # Try to correlate with market data
                self._correlate_trade_with_market_data(tx_to, tx, block_timestamp)
            else:
                # Contract interaction
                if tx_from in self.wallets:
                    self.wallets[tx_from].contract_interactions.add(tx_to)
    
    def _process_erc20_transfer(self, log: Dict[str, Any], block_timestamp: float):
        """Process ERC-20 transfer event."""
        topics = log.get("topics", [])
        if len(topics) < 3:
            return
        
        # ERC-20 Transfer: topics[1] = from, topics[2] = to
        from_addr = None
        to_addr = None
        
        if len(topics) > 1 and topics[1]:
            try:
                topic_str = str(topics[1])
                if len(topic_str) >= 40:
                    from_addr = "0x" + topic_str[-40:]
            except (TypeError, ValueError):
                pass
        
        if len(topics) > 2 and topics[2]:
            try:
                topic_str = str(topics[2])
                if len(topic_str) >= 40:
                    to_addr = "0x" + topic_str[-40:]
            except (TypeError, ValueError):
                pass
        
        token_addr = (log.get("address") or "").lower()
        
        if token_addr:
            self.erc20_tokens.add(token_addr)
        
        if from_addr and isinstance(from_addr, str):
            from_addr_lower = from_addr.lower()
            if from_addr_lower in self.wallets:
                self.wallets[from_addr_lower].add_erc20_transfer(log, block_timestamp)
        
        if to_addr and isinstance(to_addr, str):
            to_addr_lower = to_addr.lower()
            if to_addr_lower in self.wallets:
                self.wallets[to_addr_lower].add_erc20_transfer(log, block_timestamp)
    
    def _process_block(self, block: Dict[str, Any]):
        """Process a block and extract wallet activity."""
        block_number = block.get("number")
        block_timestamp = int(block.get("timestamp", "0x0"), 16) if isinstance(block.get("timestamp"), str) else 0
        
        transactions = block.get("transactions", [])
        for tx in transactions:
            if isinstance(tx, dict):
                self._process_transaction(tx, block_timestamp)
                
                # Get receipt for events
                tx_hash = tx.get("hash")
                if tx_hash:
                    receipt = self._json_rpc_request("eth_getTransactionReceipt", [tx_hash])
                    if receipt:
                        logs = receipt.get("logs", [])
                        for log in logs:
                            topics = log.get("topics", [])
                            # Check for ERC-20 Transfer event
                            if topics and len(topics) >= 3:
                                event_sig = topics[0].lower() if isinstance(topics[0], str) else ""
                                if event_sig == "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef":
                                    self._process_erc20_transfer(log, block_timestamp)
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            
            if "id" in data and "result" in data:
                self.subscription_id = data["result"]
                self._log("Subscribed to new blocks", {"subscription_id": self.subscription_id})
                return
            
            if data.get("method") == "eth_subscription" and "params" in data:
                params = data.get("params", {})
                result = params.get("result")
                if result:
                    block_number = result.get("number")
                    if block_number:
                        full_block = self._json_rpc_request("eth_getBlockByNumber", [block_number, True])
                        if full_block:
                            self._process_block(full_block)
                            self._log(f"Processed block {block_number}", {"tx_count": len(full_block.get("transactions", []))})
        except Exception as e:
            import traceback
            error_msg = f"Error processing message: {e}"
            stack_trace = traceback.format_exc()
            self._log(error_msg)
            print(f"Stack trace:\n{stack_trace}")
            if self.file_handle:
                try:
                    error_log = {
                        "type": "error",
                        "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                        "error": str(e),
                        "stack_trace": stack_trace,
                        "message_preview": message[:500] if message else None
                    }
                    self.file_handle.write(json.dumps(error_log, separators=(',', ':')) + "\n")
                    self.file_handle.flush()
                except:
                    pass
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        self._log("WebSocket connected")
        subscribe_msg = {
            "jsonrpc": "2.0",
            "method": "eth_subscribe",
            "params": ["newHeads"],
            "id": 1
        }
        ws.send(json.dumps(subscribe_msg))
    
    def _on_ws_error(self, ws, error):
        self._log(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        self._log(f"WebSocket closed: {close_status_code}")
        if self.running:
            time.sleep(2)
            self._start_websocket()
    
    def _start_websocket(self):
        """Start WebSocket connection for blockchain RPC."""
        try:
            self.ws = websocket.WebSocketApp(
                self.MAINNET_RPC_WS,
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
            self._log(f"Error starting WebSocket: {e}")
    
    def _on_api_ws_message(self, ws, message):
        """Handle Hyperliquid API WebSocket messages."""
        if NO_PUBLIC_API:
            return
        
        try:
            data = json.loads(message)
            
            with self.market_data_lock:
                channel = data.get("channel", "unknown")
                
                # Handle l2Book updates
                if channel == "l2Book":
                    coin = data.get("data", {}).get("coin")
                    if coin:
                        # Update market data
                        self.market_data['l2_book'][coin] = {
                            'data': data.get("data"),
                            'timestamp': time.time()
                        }
                        # Track statistics
                        if channel not in self.api_message_stats:
                            self.api_message_stats[channel] = {}
                        self.api_message_stats[channel][coin] = self.api_message_stats[channel].get(coin, 0) + 1
                
                # Handle bbo updates
                elif channel == "bbo":
                    coin = data.get("data", {}).get("coin")
                    if coin:
                        self.market_data['bbo'][coin] = {
                            'data': data.get("data"),
                            'timestamp': time.time()
                        }
                        # Track statistics
                        if channel not in self.api_message_stats:
                            self.api_message_stats[channel] = {}
                        self.api_message_stats[channel][coin] = self.api_message_stats[channel].get(coin, 0) + 1
                
                # Handle candle updates
                elif channel == "candle":
                    coin = data.get("data", {}).get("coin")
                    if coin:
                        self.market_data['candle'][coin] = {
                            'data': data.get("data"),
                            'timestamp': time.time()
                        }
                        # Track statistics
                        if channel not in self.api_message_stats:
                            self.api_message_stats[channel] = {}
                        self.api_message_stats[channel][coin] = self.api_message_stats[channel].get(coin, 0) + 1
                
                # Handle allMids updates
                elif channel == "allMids":
                    mids = data.get("data", {})
                    if mids:
                        self.market_data['all_mids'] = {
                            'data': mids,
                            'timestamp': time.time()
                        }
                        # Track statistics (allMids doesn't have a specific coin, count as "all")
                        if channel not in self.api_message_stats:
                            self.api_message_stats[channel] = {}
                        self.api_message_stats[channel]["all"] = self.api_message_stats[channel].get("all", 0) + 1
                        
                        # Extract available coins from allMids, but only subscribe to API_TOKENS
                        if isinstance(mids, dict):
                            # Filter to only coins in API_TOKENS
                            available_mids = {k: v for k, v in mids.items() if k in API_TOKENS}
                            new_coins = [coin for coin in available_mids.keys() if coin not in self.coins_subscribed]
                            if new_coins:
                                # Only track API_TOKENS, not all discovered coins
                                self.available_coins = [coin for coin in API_TOKENS if coin in mids.keys()]
                                self._log(f"Discovered {len(new_coins)} new coins from API_TOKENS, subscribing to all...")
                                # Subscribe to all new coins at once
                                self._subscribe_to_coins(new_coins)
                
                # Handle trades
                elif channel == "trades":
                    trades = data.get("data", [])
                    # Trades can be an array or a single object with coin field
                    if isinstance(trades, list):
                        trade_list = trades
                    elif isinstance(trades, dict) and "data" in trades:
                        trade_list = trades.get("data", [])
                    else:
                        trade_list = [trades] if trades else []
                    
                    if trade_list:
                        # Keep last 1000 trades
                        self.market_data['trades'].extend(trade_list)
                        if len(self.market_data['trades']) > 1000:
                            self.market_data['trades'] = self.market_data['trades'][-1000:]
                        
                        # Track statistics per coin
                        if channel not in self.api_message_stats:
                            self.api_message_stats[channel] = {}
                        
                        # Count trades per coin
                        for trade in trade_list:
                            if isinstance(trade, dict):
                                coin = trade.get("coin")
                                if coin:
                                    self.api_message_stats[channel][coin] = self.api_message_stats[channel].get(coin, 0) + 1
                                else:
                                    # Try to get coin from parent data structure
                                    coin = data.get("coin")
                                    if coin:
                                        self.api_message_stats[channel][coin] = self.api_message_stats[channel].get(coin, 0) + 1
                                    else:
                                        self.api_message_stats[channel]["unknown"] = self.api_message_stats[channel].get("unknown", 0) + 1
                            else:
                                self.api_message_stats[channel]["unknown"] = self.api_message_stats[channel].get("unknown", 0) + 1
                        
                        # Check if any wallets we're tracking are involved in these trades
                        for trade in trade_list:
                            if isinstance(trade, dict):
                                self._check_wallet_trade_involvement(trade)
        
        except Exception as e:
            self._log(f"Error processing API message: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_api_ws_error(self, ws, error):
        """Handle Hyperliquid API WebSocket errors."""
        self._log(f"API WebSocket error: {error}")
    
    def _on_api_ws_close(self, ws, close_status_code, close_msg):
        """Handle Hyperliquid API WebSocket close."""
        if NO_PUBLIC_API:
            return
        
        self._log(f"API WebSocket closed: {close_status_code} (total connections: {self.api_connection_count})")
        
        # Check if we've exceeded the maximum before reconnecting
        if self.api_connection_count >= self.max_api_connections:
            self._log(f"ERROR: Reached maximum API connections ({self.max_api_connections}). Not reconnecting.")
            print(f"\n[ERROR] Reached maximum API connections ({self.max_api_connections}). Stopping profiler.")
            self.running = False
            return
        
        if self.running and not NO_PUBLIC_API:
            time.sleep(2)
            self._start_api_websocket()
    
    def _get_available_coins(self) -> List[str]:
        """Get list of coins to subscribe to (only API_TOKENS)."""
        # Only subscribe to tokens in API_TOKENS list
        return [token for token in API_TOKENS]
    
    def _is_ws_connected(self) -> bool:
        """Check if WebSocket is still connected."""
        if not self.api_ws:
            return False
        try:
            if hasattr(self.api_ws, 'sock') and self.api_ws.sock:
                return self.api_ws.sock.connected
        except:
            pass
        return False
    
    def _subscribe_to_coins(self, coins: List[str]):
        """Subscribe to market data streams for given coins."""
        if not self.api_ws or not self._is_ws_connected():
            self._log("WebSocket not connected, skipping subscription")
            return
        
        subscriptions = []
        new_coins_list = []
        for coin in coins:
            if coin in self.coins_subscribed:
                continue
            
            subscriptions.extend([
                {"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}},
                {"method": "subscribe", "subscription": {"type": "bbo", "coin": coin}},
                {"method": "subscribe", "subscription": {"type": "candle", "coin": coin, "interval": "1m"}},
                {"method": "subscribe", "subscription": {"type": "trades", "coin": coin}},
            ])
            new_coins_list.append(coin)
            self.coins_subscribed.add(coin)
        
        if not subscriptions:
            return
        
        # Subscribe to all at once without delays
        successful = 0
        failed = 0
        
        self._log(f"Subscribing to {len(new_coins_list)} coins ({len(subscriptions)} total subscriptions)...")
        
        for sub in subscriptions:
            # Extract subscription details for logging
            subscription_info = sub.get("subscription", {})
            coin = subscription_info.get("coin", "unknown")
            sub_type = subscription_info.get("type", "unknown")
            
            try:
                # Send subscription immediately
                self.api_ws.send(json.dumps(sub))
                successful += 1
                
            except (OSError, ConnectionError, AttributeError, ConnectionAbortedError) as e:
                failed += 1
                # Connection error - log and continue
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["closed", "bad_length", "10054", "aborted", "reset"]):
                    self._log(f"Connection error during subscription: {type(e).__name__}")
                    self._log(f"Failed subscription details: coin='{coin}', type='{sub_type}', subscription={sub}")
                    print(f"\n[ERROR] Connection error while subscribing to: coin='{coin}', type='{sub_type}'")
                    print(f"[ERROR] Full subscription: {json.dumps(sub, indent=2)}")
                else:
                    self._log(f"Error subscribing to coin='{coin}', type='{sub_type}': {type(e).__name__}")
            except Exception as e:
                failed += 1
                self._log(f"Error subscribing to coin='{coin}', type='{sub_type}': {type(e).__name__}")
        
        self.total_subscriptions_made += successful
        self._log(f"Subscribed to {len(new_coins_list)} coins: {successful} subscriptions successful, {failed} failed")
        self._log(f"Total subscriptions made so far: {self.total_subscriptions_made}")
        self._log(f"Subscriptions: {successful} successful, {failed} failed | Total: {self.total_subscriptions_made}")
        
        # Verify connection is still alive after subscriptions
        if not self._is_ws_connected():
            self._log("WARNING: Connection lost after sending subscriptions")
        else:
            self._log("Connection verified: still connected after subscriptions")
    
    def _on_api_ws_open(self, ws):
        """Handle Hyperliquid API WebSocket open."""
        if NO_PUBLIC_API:
            self._log("Public API subscriptions disabled, closing connection")
            ws.close()
            return
        
        self.api_connection_count += 1
        self._log(f"Hyperliquid API WebSocket connected (connection #{self.api_connection_count})")
        
        # Check if we've exceeded the maximum connections
        if self.api_connection_count > self.max_api_connections:
            self._log(f"ERROR: Exceeded maximum API connections ({self.max_api_connections}). Aborting.")
            print(f"\n[ERROR] Exceeded maximum API connections ({self.max_api_connections}). Stopping profiler.")
            self.running = False
            if self.api_ws:
                self.api_ws.close()
            return
        
        # First, subscribe to allMids to get all available coins
        try:
            subscribe_allmids = {"method": "subscribe", "subscription": {"type": "allMids"}}
            ws.send(json.dumps(subscribe_allmids))
            self._log("Subscribed to allMids to discover available coins")
        except Exception as e:
            self._log(f"Error subscribing to allMids: {e}")
        
        # Get coins from API_TOKENS list
        coins = self._get_available_coins()
        if coins:
            self.available_coins = coins
            self._log(f"Subscribing to all {len(coins)} tokens from API_TOKENS list...")
            # Subscribe to all coins at once
            self._subscribe_to_coins(coins)
        else:
            self._log("No tokens in API_TOKENS list to subscribe to")
    
    def _start_api_websocket(self):
        """Start Hyperliquid API WebSocket connection."""
        if NO_PUBLIC_API:
            return
        
        try:
            self.api_ws = websocket.WebSocketApp(
                self.HYPERLIQUID_API_WS,
                on_message=self._on_api_ws_message,
                on_error=self._on_api_ws_error,
                on_close=self._on_api_ws_close,
                on_open=self._on_api_ws_open
            )
            def run_api_ws():
                self._log("API WebSocket thread started")
                self.api_ws.run_forever()
                self._log("API WebSocket thread ended")
            self.api_ws_thread = threading.Thread(target=run_api_ws, daemon=True)
            self.api_ws_thread.start()
            self._log("API WebSocket thread created")
        except Exception as e:
            self._log(f"Error starting API WebSocket: {e}")
    
    def _calculate_market_metrics(self, wallet_addr: str, trade_data: Dict[str, Any], timestamp: float):
        """Calculate market metrics for a trade and update wallet features."""
        coin = trade_data.get("coin", "").upper()
        if not coin:
            return
        
        wallet = self.wallets.get(wallet_addr.lower())
        if not wallet:
            return
        
        with self.market_data_lock:
            # Get market state at trade time
            bbo = self.market_data['bbo'].get(coin, {}).get('data', {})
            l2_book = self.market_data['l2_book'].get(coin, {}).get('data', {})
            candle = self.market_data['candle'].get(coin, {}).get('data', {})
            all_mids = self.market_data['all_mids'].get('data', {})
            
            mid_price = None
            if coin in all_mids:
                mid_price = all_mids[coin]
            elif bbo:
                bid = float(bbo.get('bid', {}).get('px', 0))
                ask = float(bbo.get('ask', {}).get('px', 0))
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
            
            if not mid_price or mid_price == 0:
                return
            
            # Get trade price and size
            trade_price = float(trade_data.get("px", 0))
            trade_size = float(trade_data.get("sz", 0))
            side = trade_data.get("side", "").upper()
            
            if trade_price == 0 or trade_size == 0:
                return
            
            # Calculate slippage (deviation from mid price)
            if side in ['B', 'BUY', 'LONG']:
                slippage = (trade_price - mid_price) / mid_price if mid_price > 0 else 0
            else:
                slippage = (mid_price - trade_price) / mid_price if mid_price > 0 else 0
            
            wallet.slippage_scores.append(abs(slippage))
            
            # Calculate market impact (trade size vs available liquidity)
            market_impact = 0.0
            if l2_book and 'levels' in l2_book:
                # Estimate available liquidity
                available_liquidity = 0.0
                levels = l2_book.get('levels', [])
                for level in levels[:5]:  # Top 5 levels
                    if side in ['B', 'BUY', 'LONG']:
                        # Buying, look at asks
                        available_liquidity += float(level.get('askSz', 0))
                    else:
                        # Selling, look at bids
                        available_liquidity += float(level.get('bidSz', 0))
                
                if available_liquidity > 0:
                    market_impact = min(1.0, trade_size / available_liquidity)
            
            wallet.market_impact_scores.append(market_impact)
            
            # Calculate momentum score (price movement direction)
            momentum_score = 0.0
            if candle and 'open' in candle and 'close' in candle:
                open_price = float(candle.get('open', 0))
                close_price = float(candle.get('close', 0))
                if open_price > 0:
                    price_change = (close_price - open_price) / open_price
                    # Positive if trading with trend
                    if (side in ['B', 'BUY', 'LONG'] and price_change > 0) or \
                       (side in ['S', 'SELL', 'SHORT'] and price_change < 0):
                        momentum_score = min(1.0, abs(price_change) * 10)
                    else:
                        momentum_score = 0.0
            
            wallet.momentum_scores.append(momentum_score)
            
            # Calculate mean reversion score (opposite of momentum)
            mean_reversion_score = 0.0
            if candle and 'open' in candle and 'close' in candle:
                open_price = float(candle.get('open', 0))
                close_price = float(candle.get('close', 0))
                if open_price > 0:
                    price_change = (close_price - open_price) / open_price
                    # Positive if trading against trend
                    if (side in ['B', 'BUY', 'LONG'] and price_change < 0) or \
                       (side in ['S', 'SELL', 'SHORT'] and price_change > 0):
                        mean_reversion_score = min(1.0, abs(price_change) * 10)
                    else:
                        mean_reversion_score = 0.0
            
            wallet.mean_reversion_scores.append(mean_reversion_score)
            
            # Calculate volatility score
            volatility_score = 0.0
            if candle and 'high' in candle and 'low' in candle:
                high = float(candle.get('high', 0))
                low = float(candle.get('low', 0))
                if mid_price > 0:
                    volatility = (high - low) / mid_price
                    volatility_score = min(1.0, volatility * 10)
            
            wallet.volatility_scores.append(volatility_score)
            
            # Calculate market maker score (order book participation)
            market_maker_score = 0.0
            order_book_participation = 0.0
            if bbo:
                spread = 0.0
                bid_px = float(bbo.get('bid', {}).get('px', 0))
                ask_px = float(bbo.get('ask', {}).get('px', 0))
                if bid_px > 0 and ask_px > 0:
                    spread = (ask_px - bid_px) / mid_price if mid_price > 0 else 0
                    # Market makers have tight spreads and consistent presence
                    if spread < 0.001:  # Very tight spread
                        market_maker_score = 1.0 - (spread * 1000)
                    else:
                        market_maker_score = max(0.0, 1.0 - (spread * 100))
            
            # Order book participation (simplified - would need more data)
            if l2_book:
                order_book_participation = 0.5  # Placeholder - would need actual order placement data
            
            wallet.market_maker_scores.append(market_maker_score)
            wallet.order_book_participation.append(order_book_participation)
            
            # Store trade with market data
            wallet.trades_with_market_data.append({
                'trade': trade_data,
                'market_state': {
                    'mid_price': mid_price,
                    'slippage': slippage,
                    'market_impact': market_impact,
                    'momentum': momentum_score,
                    'mean_reversion': mean_reversion_score,
                    'volatility': volatility_score
                },
                'timestamp': timestamp
            })
    
    def _check_wallet_trade_involvement(self, trade: Dict[str, Any]):
        """Check if any tracked wallet is involved in this trade."""
        # This is a simplified check - in reality, you'd need to match by address or other identifiers
        # For now, we'll check if the trade timestamp matches any recent wallet activity
        trade_time = trade.get('time', 0)
        if isinstance(trade_time, (int, float)):
            # Convert ms to seconds if needed
            if trade_time > 1e12:  # Likely in milliseconds
                trade_time = trade_time / 1000.0
        else:
            return
        
        current_time = time.time()
        
        # Check if any wallet had activity around this trade time (within 5 seconds)
        for wallet_addr, wallet in self.wallets.items():
            if wallet.transaction_times:
                # Check if wallet had activity near trade time
                for tx_time in wallet.transaction_times[-10:]:  # Check last 10 transactions
                    if abs(tx_time - trade_time) < 5.0:  # Within 5 seconds
                        self.wallets_in_trades.add(wallet_addr)
                        break
    
    def _correlate_trade_with_market_data(self, wallet_addr: str, tx_data: Dict[str, Any], timestamp: float):
        """Correlate an on-chain transaction with market data if it's a trade."""
        # Check if this transaction might be a trade
        # Look for ERC-20 transfers or other indicators
        # For now, we'll check market data trades stream for matching trades
        
        with self.market_data_lock:
            # Check recent trades for potential matches
            recent_trades = self.market_data['trades'][-100:]  # Last 100 trades
            
            # Try to match by timestamp proximity (within 5 seconds)
            for trade in recent_trades:
                trade_time = trade.get('time', 0)
                if isinstance(trade_time, (int, float)):
                    # Convert ms to seconds if needed
                    if trade_time > 1e12:  # Likely in milliseconds
                        trade_time = trade_time / 1000.0
                else:
                    continue
                
                if abs(trade_time - timestamp) < 5.0:  # Within 5 seconds
                    # Potential match - mark wallet as involved in trade
                    self.wallets_in_trades.add(wallet_addr)
                    # Calculate metrics
                    self._calculate_market_metrics(wallet_addr, trade, timestamp)
                    break
    
    def _train_model(self):
        """Train the MTL model on collected wallet features."""
        if len(self.wallets) < 10:
            self._log("Not enough wallets for training", {"wallet_count": len(self.wallets)})
            return
        
        self._log("Starting model training", {"wallet_count": len(self.wallets)})
        
        # Extract features
        features_list = []
        wallet_addresses = []
        for addr, wallet in self.wallets.items():
            if wallet.transactions or wallet.erc20_transfers:
                features = wallet.extract_features()
                feature_vector = [features.get(name, 0.0) for name in self.feature_names]
                features_list.append(feature_vector)
                wallet_addresses.append(addr)
        
        if len(features_list) < 10:
            self._log("Not enough wallets with activity for training")
            return
        
        # Normalize features
        features_array = np.array(features_list)
        feature_mean = features_array.mean(axis=0)
        feature_std = features_array.std(axis=0) + 1e-8
        features_normalized = (features_array - feature_mean) / feature_std
        
        # Initialize model
        if self.model is None:
            self.model = WalletMTLModel(input_dim=len(self.feature_names), hidden_dim=256).to(self.device)
        
        # Create dataset (using self-supervised learning since we don't have labels)
        dataset = WalletDataset(features_normalized.tolist())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Unsupervised training using feature reconstruction
        # In production, you'd use labeled data for supervised learning
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(5):  # Few epochs for incremental training
            total_loss = 0
            for batch in dataloader:
                features = batch['features'].to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(features)
                
                # Multi-task loss: encourage diverse outputs
                # This is unsupervised - in production use labeled data
                loss = 0
                # Encourage non-zero outputs (diversity)
                for key, output in outputs.items():
                    if key == 'style':
                        # Style should have some variance
                        loss += 0.1 * torch.mean(torch.abs(output))
                    elif key in ['risk', 'profitability', 'bot', 'sophistication']:
                        # Encourage values away from extremes
                        loss += 0.1 * torch.mean((output - 0.5) ** 2)
                    elif key == 'influence':
                        # Influence should be non-negative
                        loss += 0.1 * torch.mean(torch.relu(-output))
                
                # Regularization
                loss += 0.01 * sum(p.pow(2.0).sum() for p in self.model.parameters())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 2 == 0:
                self._log(f"Training epoch {epoch}, loss: {total_loss/len(dataloader):.4f}")
        
        # Store normalization stats
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_mean': feature_mean.tolist(),
            'feature_std': feature_std.tolist(),
            'feature_names': self.feature_names
        }, self.model_file)
        self._log("Model saved", {"file": self.model_file})
    
    def _generate_report(self):
        """Generate final report with all wallet metrics."""
        if self.model is None:
            self._log("Model not trained, cannot generate report")
            return
        
        if self.feature_mean is None or self.feature_std is None:
            self._log("Feature normalization not available, cannot generate report")
            return
        
        self._log("Generating final report", {"wallet_count": len(self.wallets)})
        
        # Extract and normalize features
        wallet_results = []
        for addr, wallet in self.wallets.items():
            if not wallet.transactions and not wallet.erc20_transfers:
                continue
            
            features = wallet.extract_features()
            feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
            
            # Normalize using saved statistics
            features_normalized = (feature_vector - self.feature_mean) / (self.feature_std + 1e-8)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(self.device)
                outputs = self.model(features_tensor)
                
                bot_probability = float(outputs['bot'].cpu().item())
                
                # Classify wallet into behavior categories
                categories = wallet.classify_wallet(features, bot_probability)
                
                result = {
                    'address': addr,
                    'style_score': outputs['style'].cpu().numpy()[0].tolist(),
                    'risk_score': float(outputs['risk'].cpu().item()),
                    'profitability_score': float(outputs['profitability'].cpu().item()),
                    'bot_probability': bot_probability,
                    'influence_score': float(outputs['influence'].cpu().item()),
                    'sophistication_score': float(outputs['sophistication'].cpu().item()),
                    'categories': categories,
                    'tx_count': features['tx_count'],
                    'age_days': features['age_days']
                }
                wallet_results.append(result)
        
        # Write report
        report_file = "final_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("WALLET PROFILING REPORT\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}\n")
            f.write(f"Total wallets analyzed: {len(wallet_results)}\n")
            f.write(f"Total RPC calls: {self.rpc_call_count}\n")
            if self.rpc_start_time:
                elapsed = time.time() - self.rpc_start_time
                f.write(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")
                f.write(f"RPC calls per second: {self.rpc_call_count/elapsed:.2f}\n")
            
            # API Statistics
            f.write("\n" + "=" * 100 + "\n")
            f.write("PUBLIC API STATISTICS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Total API connections: {self.api_connection_count}\n")
            f.write(f"Total subscriptions made: {self.total_subscriptions_made}\n")
            f.write(f"Wallets involved in API trades: {len(self.wallets_in_trades)}\n")
            if self.wallets_in_trades:
                f.write(f"Wallet addresses involved in trades: {', '.join(list(self.wallets_in_trades)[:10])}")
                if len(self.wallets_in_trades) > 10:
                    f.write(f" ... and {len(self.wallets_in_trades) - 10} more\n")
                else:
                    f.write("\n")
            f.write("\n")
            
            # Messages per channel per token
            if self.api_message_stats:
                f.write("Messages received per channel per token:\n")
                f.write("-" * 100 + "\n")
                for channel, coin_counts in sorted(self.api_message_stats.items()):
                    f.write(f"\nChannel: {channel}\n")
                    total_channel = sum(coin_counts.values())
                    f.write(f"  Total messages: {total_channel}\n")
                    for coin, count in sorted(coin_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {coin}: {count} messages\n")
            else:
                f.write("No API messages received (NO_PUBLIC_API may be enabled)\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
            
            for i, result in enumerate(sorted(wallet_results, key=lambda x: x['tx_count'], reverse=True), 1):
                f.write(f"\nWallet #{i}: {result['address']}\n")
                f.write("-" * 100 + "\n")
                
                # Categories
                if result.get('categories'):
                    category_str = ", ".join([f"{cat} ({conf:.2f})" for cat, conf in result['categories']])
                    f.write(f"Categories: {category_str}\n")
                else:
                    f.write(f"Categories: None detected\n")
                f.write("\n")
                
                # ML Metrics
                f.write(f"Trading Style Score (6D vector): {[f'{x:.4f}' for x in result['style_score']]}\n")
                f.write(f"Risk Score: {result['risk_score']:.4f}\n")
                f.write(f"Profitability Score: {result['profitability_score']:.4f}\n")
                f.write(f"Bot Probability: {result['bot_probability']:.4f}\n")
                f.write(f"Influence Score: {result['influence_score']:.4f}\n")
                f.write(f"Sophistication Score: {result['sophistication_score']:.4f}\n")
                f.write("\n")
                
                # Stats
                f.write(f"Transaction Count: {int(result['tx_count'])}\n")
                f.write(f"Age (days): {result['age_days']:.2f}\n")
                f.write("\n")
        
        self._log("Final report generated", {"file": report_file, "wallets": len(wallet_results)})
        print(f"\nFinal report saved to: {report_file}")
    
    def start(self, training_interval: int = 300):
        """Start profiling wallets."""
        # Open log file
        self.file_handle = open(self.log_file, 'w')
        self._log("Wallet profiler started")
        
        self.running = True
        self.strategy_start_time = time.time()
        
        # Start periodic status logging (every minute)
        self.status_log_thread = threading.Thread(target=self._periodic_status_log, daemon=True)
        self.status_log_thread.start()
        
        self._start_websocket()
        
        # Start Hyperliquid API WebSocket for market data (if enabled)
        if not NO_PUBLIC_API:
            self._start_api_websocket()
        else:
            self._log("Public API subscriptions disabled (NO_PUBLIC_API=True)")
        
        last_training = time.time()
        
        try:
            while self.running:
                time.sleep(1)
                
                # Periodic training
                if time.time() - last_training > training_interval:
                    if len(self.wallets) > 0:
                        self._train_model()
                    last_training = time.time()
        except KeyboardInterrupt:
            self._log("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop profiling and generate report."""
        self._log("Stopping wallet profiler")
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        if self.api_ws:
            self.api_ws.close()
        
        # Final training
        if len(self.wallets) > 0:
            self._train_model()
        
        # Try to load model if training didn't happen
        if self.model is None:
            try:
                checkpoint = torch.load(self.model_file, map_location=self.device)
                self.model = WalletMTLModel(input_dim=len(self.feature_names), hidden_dim=256).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.feature_mean = np.array(checkpoint['feature_mean'])
                self.feature_std = np.array(checkpoint['feature_std'])
                self._log("Loaded saved model", {"file": self.model_file})
            except Exception as e:
                self._log(f"Could not load model: {e}")
        
        # Generate report
        if self.model is not None and self.feature_mean is not None:
            self._generate_report()
        else:
            self._log("Cannot generate report - no model available")
        
        # Log final connection and subscription counts
        self._log(f"Total Hyperliquid API connections made: {self.api_connection_count}")
        self._log(f"Total subscriptions made: {self.total_subscriptions_made}")
        if self.api_connection_count >= self.max_api_connections:
            self._log(f"WARNING: Reached maximum API connections limit ({self.max_api_connections})")
        
        # Close log file
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None
        
        print(f"\nProfiling complete. Log saved to: {self.log_file}")
        print(f"Total Hyperliquid API connections: {self.api_connection_count}")
        print(f"Total subscriptions made: {self.total_subscriptions_made}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wallet profiling with ML")
    parser.add_argument("--log-file", type=str, default="wallet_profiling.log", help="Log file")
    parser.add_argument("--model-file", type=str, default="wallet_model.pt", help="Model file")
    parser.add_argument("--training-interval", type=int, default=300, help="Training interval in seconds")
    args = parser.parse_args()
    
    profiler = WalletProfiler(log_file=args.log_file, model_file=args.model_file)
    
    def signal_handler(sig, frame):
        profiler.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    profiler.start(training_interval=args.training_interval)


if __name__ == "__main__":
    main()
