"""
Model Runner - Stage 2: Train and run model on recorded data.

Reads data recorded by 1_record_data.py, processes wallets,
trains the model, and generates final_report.txt.
"""

import json
import time
import os
import sys
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import statistics
import math

# Import classes from wallet_profiler
sys.path.insert(0, os.path.dirname(__file__))
from wallet_profiler import (
    WalletFeatures, WalletMTLModel, WalletDataset,
    MAINNET_RPC_HTTP
)

# Constants
DATA_DIR = "recorded_data"
BLOCKS_FILE = os.path.join(DATA_DIR, "blocks.jsonl")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.jsonl")
TRADES_FILE = os.path.join(DATA_DIR, "trades.jsonl")
BBO_FILE = os.path.join(DATA_DIR, "bbo.jsonl")
L2BOOK_FILE = os.path.join(DATA_DIR, "l2book.jsonl")
MODEL_FILE = "wallet_model.pt"
FINAL_REPORT_FILE = "final_report.txt"

# ERC-20 Transfer event signature
ERC20_TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


class ModelRunner:
    """Runs model on recorded data."""
    
    def __init__(self, data_dir: str = DATA_DIR, model_file: str = MODEL_FILE):
        """Initialize the model runner."""
        self.data_dir = data_dir
        self.model_file = model_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Wallet tracking
        self.wallets: Dict[str, WalletFeatures] = {}
        
        # Model
        self.model = None
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
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Market data for processing
        self.market_data = {
            'trades': [],
            'bbo': {},
            'l2_book': {}
        }
    
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
        
        if tx_to:
            if tx_to not in self.wallets:
                self.wallets[tx_to] = WalletFeatures(tx_to)
            self.wallets[tx_to].add_transaction(tx, block_timestamp)
        
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
                    
                    if to_addr:
                        to_addr = to_addr.lower()
                        if to_addr not in self.wallets:
                            self.wallets[to_addr] = WalletFeatures(to_addr)
                        self.wallets[to_addr].add_erc20_transfer(log, block_timestamp)
    
    def _load_data(self):
        """Load all recorded data."""
        print("Loading recorded data...")
        
        # Load blocks and transactions
        blocks = self._load_jsonl(BLOCKS_FILE)
        transactions = self._load_jsonl(TRANSACTIONS_FILE)
        
        print(f"Loaded {len(blocks)} blocks, {len(transactions)} transactions")
        
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
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print("Training model...")
        num_epochs = 10
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                
                # Simple training loss (in production, use actual labels)
                loss = sum(criterion(outputs[k], torch.zeros_like(outputs[k])) for k in outputs)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        # Save model
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
            
            result = {
                'address': addr,
                'style_score': combined_outputs['style'][i].numpy().tolist(),
                'risk_score': float(combined_outputs['risk'][i]),
                'profitability_score': float(combined_outputs['profitability'][i]),
                'bot_probability': float(combined_outputs['bot'][i]),
                'influence_score': float(combined_outputs['influence'][i]),
                'sophistication_score': float(combined_outputs['sophistication'][i]),
                'tx_count': features.get('tx_count', 0),
                'age_days': features.get('age_days', 0.0)
            }
            
            # Classify wallet
            categories = wallet.classify_wallet(features, result['bot_probability'])
            result['categories'] = categories
            
            results.append(result)
        
        return results
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """Generate final report."""
        print("Generating final report...")
        
        with open(FINAL_REPORT_FILE, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WALLET PROFILING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}\n")
            f.write(f"Total wallets analyzed: {len(results)}\n\n")
            
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Wallet #{i}: {result['address']}\n")
                f.write("-" * 80 + "\n")
                
                # Categories
                categories = result.get('categories', [])
                if categories:
                    f.write("Categories: ")
                    f.write(", ".join([f"{cat} ({conf:.2f})" for cat, conf in categories[:3]]))
                    f.write("\n\n")
                
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
