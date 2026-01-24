"""
Model Runner - Stage 2: Train and run model on recorded data.

Reads data recorded by 1_record_data.py, processes wallets,
trains the model, and generates final_report.txt.
"""

import json
import time
import os
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import statistics
import math

# Constants
DATA_DIR = "recorded_data"
BLOCKS_FILE = os.path.join(DATA_DIR, "blocks.jsonl")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.jsonl")
TRADES_FILE = os.path.join(DATA_DIR, "trades.jsonl")
BBO_FILE = os.path.join(DATA_DIR, "bbo.jsonl")
L2BOOK_FILE = os.path.join(DATA_DIR, "l2book.jsonl")
CANDLES_FILE = os.path.join(DATA_DIR, "candles.jsonl")
MODEL_FILE = "wallet_model.pt"
FINAL_REPORT_FILE = "final_report.txt"
TX_COUNT_THRESHOLD = 1
SHOW_NO_CATEGORIES = False

# ERC-20 Transfer event signature
ERC20_TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Forward declarations for type hints
if False:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        WalletFeatures = None
        WalletMTLModel = None
        WalletDataset = None


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
        
        # Market data features
        self.trades_with_market_data = []
        self.market_impact_scores = []
        self.slippage_scores = []
        self.momentum_scores = []
        self.mean_reversion_scores = []
        self.volatility_scores = []
        self.market_maker_scores = []
        self.order_book_participation = []
        
        # Candle-based features (NEW)
        self.candle_contexts = []  # List of (timestamp, candle_data) tuples
        self.candle_volatilities = []  # Volatility during active periods
        self.candle_momentums = []  # Price momentum during active periods
        self.candle_trends = []  # Trend strength during active periods
        self.candle_volumes = []  # Volume patterns during active periods
    
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
    
    def add_candle_context(self, timestamp: float, candle_data: Dict[str, Any]):
        """Add candle context for a transaction timestamp."""
        self.candle_contexts.append((timestamp, candle_data))
    
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
        
        # Size entropy
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
        
        # CANDLE-BASED FEATURES (NEW)
        if self.candle_contexts:
            # Extract candle metrics
            volatilities = []
            momentums = []
            trends = []
            volumes = []
            
            for _, candle in self.candle_contexts:
                try:
                    o = float(candle.get('o', 0))
                    h = float(candle.get('h', 0))
                    l = float(candle.get('l', 0))
                    c = float(candle.get('c', 0))
                    v = float(candle.get('v', 0))
                    
                    if o > 0:
                        # Volatility: (high - low) / open (normalized range)
                        volatility = (h - l) / o if o > 0 else 0.0
                        volatilities.append(volatility)
                        
                        # Momentum: (close - open) / open (price change)
                        momentum = (c - o) / o if o > 0 else 0.0
                        momentums.append(momentum)
                        
                        # Trend strength: |close - open| / (high - low) (how much of the range was used)
                        range_size = h - l if h > l else 1.0
                        trend = abs(c - o) / range_size if range_size > 0 else 0.0
                        trends.append(trend)
                        
                        # Volume (normalized)
                        volumes.append(v)
                except (ValueError, TypeError):
                    continue
            
            # Aggregate candle features
            if volatilities:
                features['avg_candle_volatility'] = statistics.mean(volatilities)
                features['max_candle_volatility'] = max(volatilities)
                features['volatility_std'] = statistics.stdev(volatilities) if len(volatilities) > 1 else 0.0
            else:
                features['avg_candle_volatility'] = 0.0
                features['max_candle_volatility'] = 0.0
                features['volatility_std'] = 0.0
            
            if momentums:
                features['avg_candle_momentum'] = statistics.mean(momentums)
                features['momentum_std'] = statistics.stdev(momentums) if len(momentums) > 1 else 0.0
                # Positive momentum ratio (how often price went up)
                positive_momentum = sum(1 for m in momentums if m > 0)
                features['positive_momentum_ratio'] = positive_momentum / len(momentums)
            else:
                features['avg_candle_momentum'] = 0.0
                features['momentum_std'] = 0.0
                features['positive_momentum_ratio'] = 0.5
            
            if trends:
                features['avg_trend_strength'] = statistics.mean(trends)
            else:
                features['avg_trend_strength'] = 0.0
            
            if volumes:
                features['avg_candle_volume'] = statistics.mean(volumes)
                features['max_candle_volume'] = max(volumes)
                features['volume_std'] = statistics.stdev(volumes) if len(volumes) > 1 else 0.0
            else:
                features['avg_candle_volume'] = 0.0
                features['max_candle_volume'] = 0.0
                features['volume_std'] = 0.0
        else:
            # No candle context
            features['avg_candle_volatility'] = 0.0
            features['max_candle_volatility'] = 0.0
            features['volatility_std'] = 0.0
            features['avg_candle_momentum'] = 0.0
            features['momentum_std'] = 0.0
            features['positive_momentum_ratio'] = 0.5
            features['avg_trend_strength'] = 0.0
            features['avg_candle_volume'] = 0.0
            features['max_candle_volume'] = 0.0
            features['volume_std'] = 0.0
        
        return features
    
    def classify_wallet(self, features: Dict[str, float], bot_probability: float = 0.0,
                      risk_score: float = 0.0, profitability_score: float = 0.0,
                      sophistication_score: float = 0.0, influence_score: float = 0.0,
                      style_score: List[float] = None) -> List[tuple]:
        """Classify wallet into behavior categories using both features and ML scores."""
        categories = []
        
        # Normalize style_score if provided
        style_vector = style_score if style_score and len(style_score) == 6 else [0.0] * 6
        
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
        
        # Candle-based features
        avg_volatility = features.get('avg_candle_volatility', 0.0)
        avg_momentum = features.get('avg_candle_momentum', 0.0)
        positive_momentum_ratio = features.get('positive_momentum_ratio', 0.5)
        avg_trend_strength = features.get('avg_trend_strength', 0.0)
        avg_volume = features.get('avg_candle_volume', 0.0)
        
        # ML Confidence: Use average of key ML scores as overall confidence
        # This is used to boost categories, but rule-based categories still work without ML
        ml_confidence = (bot_probability + risk_score + profitability_score + sophistication_score) / 4.0
        # Normalize: if all scores are very low, ML might not be trained yet, so don't penalize
        use_ml_boost = ml_confidence > 0.1  # Only use ML boost if there's some signal
        
        # Bot classification (ML-driven)
        if bot_probability > 0.7:
            # Boost confidence if sophistication is also high (sophisticated bots)
            confidence = min(1.0, bot_probability * 0.8 + sophistication_score * 0.2)
            categories.append(("Bot", confidence))
        elif bot_probability > 0.5:
            confidence = min(0.9, bot_probability * 0.7 + sophistication_score * 0.1)
            categories.append(("Possible Bot", confidence))
        
        # Scalper: High frequency, small sizes, high burstiness
        # Enhanced with ML: High sophistication + high risk suggests sophisticated scalping
        if tx_per_day > 50 and size_entropy < 0.4:
            base_confidence = min(1.0, (tx_per_day / 100.0) * (1.0 - size_entropy))
            # Boost if ML suggests sophisticated high-frequency trading
            ml_boost = (sophistication_score * 0.3 + risk_score * 0.2) if use_ml_boost else 0.0
            confidence = min(1.0, base_confidence * 0.7 + ml_boost)
            categories.append(("Scalper", confidence))
        elif tx_per_day > 30 and size_entropy < 0.5:
            ml_boost = (sophistication_score * 0.2) if use_ml_boost else 0.0
            confidence = min(0.85, 0.6 + ml_boost)
            categories.append(("Possible Scalper", confidence))
        
        # HODLer: Very low frequency, old wallet
        if tx_per_day < 0.1 and age_days > 30:
            confidence = min(1.0, (30.0 / max(age_days, 1.0)) * (0.1 / max(tx_per_day, 0.01)))
            categories.append(("HODLer", confidence))
        elif tx_per_day < 0.5 and age_days > 60:
            categories.append(("Possible HODLer", 0.6))
        
        # Whale: Large transaction values
        # Require both high single transaction AND high total volume to be a true whale
        # Also require minimum transaction count to avoid one-off large transactions
        if tx_count >= 5:
            if (max_value > 50000.0 and total_value > 100000.0) or total_value > 500000.0:
                # True whale: very large single transactions AND high total volume, OR extremely high total volume
                confidence = min(1.0, (max_value / 500000.0) * 0.5 + (total_value / 5000000.0) * 0.5)
                categories.append(("Whale", min(1.0, confidence)))
            elif (max_value > 10000.0 and total_value > 50000.0) or total_value > 200000.0:
                # Possible whale: large transactions and significant total volume, OR very high total volume
                confidence = min(0.85, (max_value / 100000.0) * 0.4 + (total_value / 1000000.0) * 0.4)
                categories.append(("Possible Whale", confidence))
        
        # Token Collector: High ERC-20 activity, many tokens
        if unique_tokens > 10 and erc20_count > tx_count * 2:
            confidence = min(1.0, (unique_tokens / 50.0) * (erc20_count / max(tx_count * 3, 1.0)))
            categories.append(("Token Collector", confidence))
        elif unique_tokens > 5 and erc20_count > tx_count:
            categories.append(("Possible Token Collector", 0.6))
        
        # Active Trader: Moderate-high frequency
        # ML-enhanced: Use profitability and sophistication to gauge trader quality
        if 1.0 <= tx_per_day <= 50 and burstiness > 0.3:
            base_confidence = min(1.0, tx_per_day / 50.0)
            # Boost if profitable and sophisticated
            ml_boost = (profitability_score * 0.3 + sophistication_score * 0.2) if use_ml_boost else 0.0
            confidence = min(1.0, base_confidence * 0.6 + ml_boost)
            categories.append(("Active Trader", confidence))
        elif 0.5 <= tx_per_day <= 30:
            ml_boost = (profitability_score * 0.2) if ml_confidence > 0.3 else 0.0
            confidence = min(0.8, 0.6 + ml_boost)
            categories.append(("Moderate Trader", confidence))
        
        # Arbitrageur: Multiple tokens, high flip rate
        # ML-enhanced: High sophistication + profitability suggests successful arbitrage
        if unique_tokens > 5 and flip_rate > 0.5:
            base_confidence = min(1.0, (unique_tokens / 20.0) * flip_rate)
            # Boost if profitable and sophisticated (arbitrage requires both)
            ml_boost = (profitability_score * 0.4 + sophistication_score * 0.3) if use_ml_boost else 0.0
            confidence = min(1.0, base_confidence * 0.5 + ml_boost)
            categories.append(("Arbitrageur", confidence))
        elif unique_tokens > 3 and flip_rate > 0.3:
            ml_boost = (profitability_score * 0.3 + sophistication_score * 0.2) if use_ml_boost else 0.0
            confidence = min(0.85, 0.6 + ml_boost)
            categories.append(("Possible Arbitrageur", confidence))
        
        # ML-Driven Strategy Categories using Trading Style Vector
        # Style vector dimensions might represent: [momentum, mean_reversion, volatility, market_making, trend_following, contrarian]
        style_momentum = style_vector[0] if len(style_vector) > 0 else 0.0
        style_mean_reversion = style_vector[1] if len(style_vector) > 1 else 0.0
        style_volatility = style_vector[2] if len(style_vector) > 2 else 0.0
        
        # Volatility Trader - ML-enhanced with style vector
        if avg_volatility > 0.02 and tx_count > 5:
            base_confidence = min(1.0, (avg_volatility / 0.05) * (tx_count / 20.0))
            # Use style vector to confirm volatility trading preference
            style_boost = max(0.0, style_volatility) * 0.3 if use_ml_boost else 0.0
            confidence = min(1.0, base_confidence * 0.7 + style_boost)
            categories.append(("Volatility Trader", confidence))
        elif avg_volatility > 0.01:
            style_boost = max(0.0, style_volatility) * 0.2 if use_ml_boost else 0.0
            confidence = min(0.85, 0.6 + style_boost)
            categories.append(("Possible Volatility Trader", confidence))
        
        # Momentum Follower - ML-enhanced with style vector
        if positive_momentum_ratio > 0.7 and avg_momentum > 0.001 and tx_count > 5:
            base_confidence = min(1.0, positive_momentum_ratio * (avg_momentum / 0.01))
            # Use style vector to confirm momentum preference
            style_boost = max(0.0, style_momentum) * 0.3 if ml_confidence > 0.3 else 0.0
            ml_boost = profitability_score * 0.2 if ml_confidence > 0.3 else 0.0
            confidence = min(1.0, base_confidence * 0.5 + style_boost + ml_boost)
            categories.append(("Momentum Follower", confidence))
        elif positive_momentum_ratio > 0.6:
            style_boost = max(0.0, style_momentum) * 0.2 if use_ml_boost else 0.0
            confidence = min(0.85, 0.6 + style_boost)
            categories.append(("Possible Momentum Follower", confidence))
        
        # Contrarian Trader - ML-enhanced with style vector
        if positive_momentum_ratio < 0.3 and avg_momentum < -0.001 and tx_count > 5:
            base_confidence = min(1.0, (1.0 - positive_momentum_ratio) * (abs(avg_momentum) / 0.01))
            # Use style vector to confirm contrarian/mean reversion preference
            style_boost = max(0.0, style_mean_reversion) * 0.3 if ml_confidence > 0.3 else 0.0
            ml_boost = profitability_score * 0.2 if ml_confidence > 0.3 else 0.0
            confidence = min(1.0, base_confidence * 0.5 + style_boost + ml_boost)
            categories.append(("Contrarian Trader", confidence))
        elif positive_momentum_ratio < 0.4:
            style_boost = max(0.0, style_mean_reversion) * 0.2 if use_ml_boost else 0.0
            confidence = min(0.85, 0.6 + style_boost)
            categories.append(("Possible Contrarian Trader", confidence))
        
        # High Volume Trader - ML-enhanced
        if avg_volume > 100.0 and tx_count > 5:
            base_confidence = min(1.0, (avg_volume / 500.0) * (tx_count / 20.0))
            # Boost if profitable (high volume traders should be profitable)
            ml_boost = profitability_score * 0.3 if ml_confidence > 0.3 else 0.0
            confidence = min(1.0, base_confidence * 0.7 + ml_boost)
            categories.append(("High Volume Trader", confidence))
        
        # NEW: ML-Driven Categories based purely on ML scores
        # These work even with low ML scores - they just need to be above baseline
        
        # Profitable Trader - High profitability score (lowered thresholds)
        if profitability_score > 0.6 and tx_count >= 3:
            # Boost confidence with sophistication (profitable + sophisticated = skilled trader)
            confidence = min(1.0, profitability_score * 0.7 + sophistication_score * 0.3)
            categories.append(("Profitable Trader", confidence))
        elif profitability_score > 0.4 and tx_count >= 3:
            confidence = min(0.85, profitability_score * 0.6 + sophistication_score * 0.2)
            categories.append(("Possibly Profitable Trader", confidence))
        
        # High Risk Trader - High risk score (lowered thresholds)
        if risk_score > 0.6 and tx_count >= 3:
            # High risk + high sophistication might indicate sophisticated risk-taking
            confidence = min(1.0, risk_score * 0.7 + sophistication_score * 0.2)
            categories.append(("High Risk Trader", confidence))
        elif risk_score > 0.4 and tx_count >= 3:
            confidence = min(0.85, risk_score * 0.6)
            categories.append(("Moderate Risk Trader", confidence))
        
        # Sophisticated Trader - High sophistication score (lowered thresholds)
        if sophistication_score > 0.6 and tx_count >= 5:
            # Combine with profitability for true sophistication
            confidence = min(1.0, sophistication_score * 0.6 + profitability_score * 0.3 + risk_score * 0.1)
            categories.append(("Sophisticated Trader", confidence))
        elif sophistication_score > 0.4 and tx_count >= 3:
            confidence = min(0.85, sophistication_score * 0.7)
            categories.append(("Possibly Sophisticated Trader", confidence))
        
        # Influential Trader - High influence score (lowered thresholds)
        if influence_score > 3.0 and tx_count >= 5:
            # High influence suggests market maker or large trader
            confidence = min(1.0, min(1.0, influence_score / 50.0) * 0.7 + sophistication_score * 0.3)
            categories.append(("Influential Trader", confidence))
        elif influence_score > 1.0 and tx_count >= 3:
            confidence = min(0.85, min(1.0, influence_score / 20.0) * 0.6)
            categories.append(("Possibly Influential Trader", confidence))
        
        # Don't filter categories - let rule-based categories work even without ML
        # ML is used as a boost, not a requirement
        
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
            nn.Tanh()
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.profitability_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.bot_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.influence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        
        self.sophistication_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
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


class ModelRunner:
    """Runs model on recorded data."""
    
    def __init__(self, data_dir: str = DATA_DIR, model_file: str = MODEL_FILE):
        """Initialize the model runner."""
        self.data_dir = data_dir
        self.model_file = model_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Wallet tracking
        self.wallets: Dict[str, 'WalletFeatures'] = {}
        
        # Model
        self.model = None
        self.feature_names = [
            'tx_count', 'erc20_count', 'unique_tokens', 'unique_addresses',
            'age_days', 'tx_per_day', 'burstiness', 'hour_entropy',
            'total_value', 'avg_value', 'max_value', 'value_std',
            'erc20_volume', 'long_ratio', 'flip_rate',
            'size_entropy', 'contract_interaction_count',
            'avg_market_impact', 'max_market_impact', 'avg_slippage', 'slippage_std',
            'momentum_score', 'mean_reversion_score', 'volatility_trading_score',
            'market_maker_score', 'order_book_participation',
            # NEW: Candle-based features
            'avg_candle_volatility', 'max_candle_volatility', 'volatility_std',
            'avg_candle_momentum', 'momentum_std', 'positive_momentum_ratio',
            'avg_trend_strength', 'avg_candle_volume', 'max_candle_volume', 'volume_std'
        ]
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Market data for processing
        self.market_data = {
            'trades': [],
            'bbo': {},
            'l2_book': {},
            'candles': []  # NEW: Store candles indexed by timestamp
        }
        
        # Candle index: (coin, timestamp_ms) -> candle_data
        self.candle_index: Dict[tuple, Dict[str, Any]] = {}
    
    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Load JSON Lines file."""
        data = []
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return data
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            data.append(record.get("data", record))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
        
        return data
    
    def _process_transaction(self, tx: Dict[str, Any], block_timestamp: float):
        """Process a transaction and update wallet features."""
        tx_from = (tx.get("from") or "").lower()
        tx_to = (tx.get("to") or "").lower()
        
        if tx_from:
            if tx_from not in self.wallets:
                self.wallets[tx_from] = WalletFeatures(tx_from)
            self.wallets[tx_from].add_transaction(tx, block_timestamp)
            
            # Match candle to transaction timestamp
            self._match_candle_to_transaction(tx_from, block_timestamp)
        
        if tx_to:
            if tx_to not in self.wallets:
                self.wallets[tx_to] = WalletFeatures(tx_to)
            self.wallets[tx_to].add_transaction(tx, block_timestamp)
            
            # Match candle to transaction timestamp
            self._match_candle_to_transaction(tx_to, block_timestamp)
        
        # Process ERC-20 transfers from logs
        receipt = tx.get("receipt", {})
        logs = receipt.get("logs", [])
        for log in logs:
            topics = log.get("topics", [])
            if len(topics) > 0 and topics[0].lower() == ERC20_TRANSFER_SIG.lower():
                # ERC-20 transfer
                if len(topics) >= 3:
                    from_addr = "0x" + topics[1][-40:] if len(topics[1]) >= 42 else ""
                    to_addr = "0x" + topics[2][-40:] if len(topics[2]) >= 42 else ""
                    
                    if from_addr:
                        from_addr = from_addr.lower()
                        if from_addr not in self.wallets:
                            self.wallets[from_addr] = WalletFeatures(from_addr)
                        self.wallets[from_addr].add_erc20_transfer(log, block_timestamp)
                        self._match_candle_to_transaction(from_addr, block_timestamp)
                    
                    if to_addr:
                        to_addr = to_addr.lower()
                        if to_addr not in self.wallets:
                            self.wallets[to_addr] = WalletFeatures(to_addr)
                        self.wallets[to_addr].add_erc20_transfer(log, block_timestamp)
                        self._match_candle_to_transaction(to_addr, block_timestamp)
    
    def _match_candle_to_transaction(self, address: str, tx_timestamp: float):
        """Match the closest candle to a transaction timestamp."""
        if not self.candle_index:
            return
        
        # Convert transaction timestamp to milliseconds
        tx_timestamp_ms = int(tx_timestamp * 1000)
        
        # Find the closest candle (candle.t is the open time in ms)
        best_candle = None
        best_diff = float('inf')
        
        for (coin, candle_t), candle_data in self.candle_index.items():
            # Check if transaction is within candle period (t to T)
            candle_T = candle_data.get('T', candle_t + 60000)  # Default 1 minute if T missing
            if candle_t <= tx_timestamp_ms <= candle_T:
                # Transaction is within this candle period
                diff = abs(tx_timestamp_ms - candle_t)
                if diff < best_diff:
                    best_diff = diff
                    best_candle = candle_data
        
        if best_candle:
            wallet = self.wallets[address]
            wallet.add_candle_context(tx_timestamp, best_candle)
    
    def _load_data(self):
        """Load all recorded data."""
        print("Loading recorded data...")
        
        # Load blocks and transactions
        blocks = self._load_jsonl(BLOCKS_FILE)
        transactions = self._load_jsonl(TRANSACTIONS_FILE)
        
        print(f"Loaded {len(blocks)} blocks, {len(transactions)} transactions")
        
        # Load candles FIRST to build index
        candles = self._load_jsonl(CANDLES_FILE)
        print(f"Loaded {len(candles)} candles")
        
        # Index candles by (coin, timestamp)
        for candle in candles:
            coin = candle.get('s', candle.get('coin', '')).upper()
            candle_t = candle.get('t')  # Open time in milliseconds
            if coin and candle_t:
                self.candle_index[(coin, candle_t)] = candle
        
        print(f"Indexed {len(self.candle_index)} candles")
        
        # Process transactions
        for tx in transactions:
            # Try to get timestamp from block or use current time
            block_timestamp = tx.get("timestamp")
            if not block_timestamp:
                # Try to get from block number
                block_number = tx.get("blockNumber")
                if block_number:
                    # Estimate timestamp (rough approximation)
                    block_timestamp = time.time() - (int(block_number, 16) if isinstance(block_number, str) else block_number) * 2
                else:
                    block_timestamp = time.time()
            
            self._process_transaction(tx, block_timestamp)
        
        # Load market data (trades, BBO, L2Book)
        trades = self._load_jsonl(TRADES_FILE)
        bbo_data = self._load_jsonl(BBO_FILE)
        l2book_data = self._load_jsonl(L2BOOK_FILE)
        
        print(f"Loaded {len(trades)} trades, {len(bbo_data)} BBO updates, {len(l2book_data)} L2Book updates")
        
        # Store market data for later processing
        self.market_data['trades'] = trades
        for bbo in bbo_data:
            coin = bbo.get("coin", "").upper()
            if coin:
                self.market_data['bbo'][coin] = bbo
        
        for l2book in l2book_data:
            coin = l2book.get("coin", "").upper()
            if coin:
                self.market_data['l2_book'][coin] = l2book
        
        print(f"Processed {len(self.wallets)} unique wallets")
    
    def _normalize_features(self, features_list: List[np.ndarray]) -> np.ndarray:
        """Normalize features."""
        if not features_list:
            return np.array([])
        
        features_array = np.array(features_list)
        
        # Calculate mean and std
        self.feature_mean = np.mean(features_array, axis=0)
        self.feature_std = np.std(features_array, axis=0)
        
        # Avoid division by zero
        self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        
        # Normalize
        normalized = (features_array - self.feature_mean) / self.feature_std
        
        return normalized
    
    def _train_model(self):
        """Train the model on wallet features."""
        if len(self.wallets) == 0:
            print("No wallets to train on")
            return
        
        print("Extracting features from wallets...")
        features_list = []
        for wallet in self.wallets.values():
            features = wallet.extract_features()
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            features_list.append(feature_vector)
        
        if not features_list:
            print("No features extracted")
            return
        
        # Normalize features
        features_array = self._normalize_features(features_list)
        
        # Create dataset
        dataset = WalletDataset(features_array.tolist())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        self.model = WalletMTLModel(input_dim=len(self.feature_names))
        self.model.to(self.device)
        
        # Training setup with improved configuration
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        print("Training model...")
        num_epochs = 20  # Increased epochs for better learning
        best_loss = float('inf')
        patience_counter = 0
        patience = 5  # Early stopping patience
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            self.model.train()  # Set to training mode
            
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                
                # Weighted loss: Different tasks have different importance
                # Bot detection and profitability are most important
                task_weights = {
                    'style': 0.1,
                    'risk': 0.15,
                    'profitability': 0.25,  # Higher weight
                    'bot': 0.25,  # Higher weight
                    'influence': 0.1,
                    'sophistication': 0.15
                }
                
                # Self-supervised learning: Learn to reconstruct features through shared encoder
                # This helps the model learn meaningful representations
                loss = sum(task_weights.get(k, 0.1) * criterion(outputs[k], torch.zeros_like(outputs[k])) for k in outputs)
                
                # Add regularization: Encourage diverse outputs (not all zeros)
                # This prevents the model from collapsing to trivial solutions
                output_diversity = sum(torch.std(outputs[k], dim=0).mean() for k in outputs)
                loss = loss - 0.01 * output_diversity  # Encourage diversity
                
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Print progress every epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.model_file)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                    # Load best model
                    self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
                    break
        
        # Final save
        torch.save(self.model.state_dict(), self.model_file)
        print(f"Model saved to {self.model_file}")
    
    def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for all wallets."""
        if not self.model:
            print("Model not trained. Loading existing model...")
            try:
                self.model = WalletMTLModel(input_dim=len(self.feature_names))
                self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
                return []
        
        print("Generating predictions...")
        results = []
        
        # Extract features
        features_list = []
        wallet_addresses = []
        for addr, wallet in self.wallets.items():
            features = wallet.extract_features()
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            features_list.append(feature_vector)
            wallet_addresses.append(addr)
        
        if not features_list:
            return results
        
        # Normalize
        if self.feature_mean is None or self.feature_std is None:
            features_array = self._normalize_features(features_list)
        else:
            features_array = np.array(features_list)
            features_array = (features_array - self.feature_mean) / self.feature_std
        
        # Create dataset and dataloader
        dataset = WalletDataset(features_array.tolist())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Generate predictions
        self.model.eval()
        all_outputs = []
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                outputs = self.model(features)
                all_outputs.append(outputs)
        
        # Combine outputs
        combined_outputs = {}
        for key in ['style', 'risk', 'profitability', 'bot', 'influence', 'sophistication']:
            combined_outputs[key] = torch.cat([out[key].cpu() for out in all_outputs], dim=0)
        
        # Create results
        for i, addr in enumerate(wallet_addresses):
            wallet = self.wallets[addr]
            features = wallet.extract_features()
            tx_count = features.get('tx_count', 0)
            
            # Filter by transaction count threshold
            if tx_count < TX_COUNT_THRESHOLD:
                continue
            
            result = {
                'address': addr,
                'style_score': combined_outputs['style'][i].numpy().tolist(),
                'risk_score': float(combined_outputs['risk'][i]),
                'profitability_score': float(combined_outputs['profitability'][i]),
                'bot_probability': float(combined_outputs['bot'][i]),
                'influence_score': float(combined_outputs['influence'][i]),
                'sophistication_score': float(combined_outputs['sophistication'][i]),
                'tx_count': tx_count
            }
            
            # Classify wallet with ML scores
            categories = wallet.classify_wallet(
                features,
                bot_probability=result['bot_probability'],
                risk_score=result['risk_score'],
                profitability_score=result['profitability_score'],
                sophistication_score=result['sophistication_score'],
                influence_score=result['influence_score'],
                style_score=result['style_score']
            )
            result['categories'] = categories
            
            results.append(result)
        
        return results
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """Generate final report."""
        print("Generating final report...")
        
        # Filter results if SHOW_NO_CATEGORIES is False
        filtered_results = results
        if not SHOW_NO_CATEGORIES:
            filtered_results = [r for r in results if r.get('categories', [])]
        
        with open(FINAL_REPORT_FILE, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WALLET PROFILING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}\n")
            f.write(f"Total wallets analyzed: {len(results)}\n")
            if not SHOW_NO_CATEGORIES:
                f.write(f"Wallets with categories: {len(filtered_results)}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(filtered_results, 1):
                f.write(f"Wallet #{i}: {result['address']}\n")
                f.write("-" * 80 + "\n")
                
                # Categories - ALWAYS SHOW (priority)
                categories = result.get('categories', [])
                if categories:
                    f.write("Categories: ")
                    # Show all categories, not just top 3
                    f.write(", ".join([f"{cat} ({conf:.2f})" for cat, conf in categories]))
                    f.write("\n\n")
                else:
                    f.write("Categories: None\n\n")
                
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
                f.write("\n")
        
        print(f"Final report saved to: {FINAL_REPORT_FILE}")
    
    def run(self):
        """Run the complete pipeline."""
        print("=" * 80)
        print("MODEL RUNNER - STAGE 2")
        print("=" * 80)
        
        # Load data
        self._load_data()
        
        if len(self.wallets) == 0:
            print("No wallets found in recorded data")
            return
        
        # Train model
        self._train_model()
        
        # Generate predictions
        results = self._generate_predictions()
        
        if not results:
            print("No predictions generated")
            return
        
        # Generate report
        self._generate_report(results)
        
        print("=" * 80)
        print("Pipeline complete!")
        print("=" * 80)


def main():
    """Main entry point."""
    runner = ModelRunner()
    runner.run()


if __name__ == "__main__":
    main()
