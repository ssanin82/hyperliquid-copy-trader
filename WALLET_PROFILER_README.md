# Wallet Profiler - ML-Based Wallet Analysis

## Overview

The Wallet Profiler uses PyTorch multi-task learning to analyze blockchain wallets and derive 6 key metrics:
- **Trading Style Score**: 6D vector representing trading behavior patterns
- **Risk Score**: 0-1 score indicating risk level
- **Profitability Score**: Expected profitability (0-1)
- **Bot Probability**: Likelihood of being automated (0-1)
- **Influence Score**: How many wallets follow this one
- **Sophistication Score**: Complexity of trading strategies (0-1)

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install the package:

```bash
pip install -e .
```

## How to Run

### Basic Usage

```bash
python -m hypertrack.wallet_profiler
```

Or use the console script (after installation):

```bash
hypertrack-profile
```

### With Options

```bash
# Custom log file
hypertrack-profile --log-file my_profiling.log

# Custom model file
hypertrack-profile --model-file my_model.pt

# Custom training interval (in seconds)
hypertrack-profile --training-interval 600  # Train every 10 minutes
```

### Command Line Options

- `--log-file`: File to save all profiling data (default: `wallet_profiling.log`)
- `--model-file`: File to save/load the PyTorch model (default: `wallet_model.pt`)
- `--training-interval`: How often to retrain the model in seconds (default: 300 = 5 minutes)

## How It Works

1. **Blockchain Subscription**: Subscribes to new blocks via `eth_subscribe("newHeads")`
2. **Transaction Processing**: Processes all transactions in each block
3. **Wallet Detection**: Automatically detects EOA (wallet) addresses
4. **ERC-20 Detection**: Detects ERC-20 token transfers
5. **Feature Extraction**: Extracts 19 features per wallet:
   - Transaction counts, frequencies, patterns
   - Temporal features (burstiness, hour entropy)
   - Value features (total, average, max, std)
   - ERC-20 activity
   - Directional patterns
   - Network interactions
6. **ML Training**: Trains a multi-task neural network periodically
7. **Report Generation**: Generates `final_report.txt` with all metrics

## Training Duration Recommendations

### Minimum Training Time
- **1-2 hours**: For initial model training and basic profiling
- **Minimum wallets needed**: At least 50-100 wallets with activity

### Recommended Training Time
- **4-8 hours**: For more accurate metrics and better model convergence
- **Ideal**: 24+ hours for comprehensive profiling of many wallets

### Factors Affecting Training Time

1. **Blockchain Activity**: More active chains = more data faster
2. **Wallet Diversity**: Need diverse wallet behaviors for good training
3. **Model Convergence**: Model improves with more training data
4. **RPC Rate Limits**: Alchemy RPC has rate limits (check your plan)

### What Happens During Training

- **First 5 minutes**: Collecting initial wallet data
- **Every 5 minutes (default)**: Model retrains on accumulated data
- **On stop (Ctrl+C)**: Final training + report generation

## Output Files

### `wallet_profiling.log`
JSON Lines format with:
- All detected wallets
- Transaction processing events
- Training progress
- Model checkpoints
- RPC call statistics

### `wallet_model.pt`
PyTorch model checkpoint containing:
- Model weights
- Feature normalization statistics
- Feature names

### `final_report.txt`
Human-readable report with:
- Summary statistics
- Per-wallet metrics:
  - Trading style score (6D vector)
  - Risk score
  - Profitability score
  - Bot probability
  - Influence score
  - Sophistication score
  - Transaction count
  - Age in days

## Example Output

```
================================================================================
WALLET PROFILING REPORT
================================================================================
Generated: 2026-01-21T12:00:00Z
Total wallets analyzed: 150
Total RPC calls: 5000
Total time: 3600.00 seconds (60.00 minutes)
RPC calls per second: 1.39
================================================================================

Wallet #1: 0x1234...
--------------------------------------------------------------------------------
Trading Style Score (6D vector): ['0.2341', '-0.1234', '0.5678', '0.1234', '-0.2341', '0.3456']
Risk Score: 0.7234
Profitability Score: 0.8123
Bot Probability: 0.1234
Influence Score: 5.2341
Sophistication Score: 0.6789
Transaction Count: 1250
Age (days): 45.23
```

## Important Notes

1. **No Private Data**: Does NOT subscribe to `userEvents` or `userTrades` (those are private)
2. **Public Blockchain Only**: Only uses publicly available blockchain data
3. **EOA Detection**: Uses `eth_getCode` to distinguish wallets from contracts
4. **ERC-20 Detection**: Detects Transfer events to identify token interactions
5. **Incremental Learning**: Model trains periodically as new data arrives
6. **GPU Support**: Automatically uses GPU if available (CUDA)

## Troubleshooting

### "Not enough wallets for training"
- Wait longer to collect more wallet data
- Check if blockchain is active
- Verify RPC connection

### "Model not trained, cannot generate report"
- Ensure you've run for at least 5-10 minutes
- Check that wallets are being detected (check log file)
- Verify at least 10 wallets have transactions

### High RPC call rate
- Adjust training interval to reduce frequency
- Check Alchemy rate limits
- Consider using a higher-tier RPC plan

## Stopping the Profiler

Press `Ctrl+C` to gracefully stop:
1. Final model training
2. Report generation
3. Log file closure

All data is saved automatically.
