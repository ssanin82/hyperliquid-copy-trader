<img src="paper.png" alt="paper" style="width: 50%; max-width: 100%;">

# Hyperliquid Copy Trader - Wallet Profiling System

A Python package for tracking Hyperliquid trading activity, recording market data, and using machine learning to detect and classify profitable traders using real-time data streams.

## Table of Contents

- [Overview](#overview)
- [Why WebSocket Streams vs Blockchain Events?](#why-websocket-streams-vs-blockchain-events)
- [Features](#features)
- [Installation](#installation)
  - [From PyPI (when published)](#from-pypi-when-published)
  - [From source](#from-source)
- [Usage](#usage)
  - [Stage 1: Data Collection](#stage-1-data-collection)
  - [Stage 2: Model Training & Classification](#stage-2-model-training--classification)
  - [Stage 3: Visualization](#stage-3-visualization)
- [Machine Learning Model](#machine-learning-model)
  - [Model Architecture: Multi-Task Learning (MTL)](#model-architecture-multi-task-learning-mtl)
  - [Training Process](#training-process)
- [Feature Engineering](#feature-engineering)
- [Wallet Classification Categories](#wallet-classification-categories)
  - [Classification Logic](#classification-logic)
  - [Why Some Wallets Have No Categories](#why-some-wallets-have-no-categories)
- [Output Format](#output-format)
- [Challenges and Solutions in Wallet Categorization](#challenges-and-solutions-in-wallet-categorization)
- [Three-Stage Architecture](#three-stage-architecture)
- [API Endpoints](#api-endpoints)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Important Notes](#important-notes)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview

This system consists of three main stages:

1. **Data Collection (`1_record_data.py`)**: Subscribes to blockchain events and Hyperliquid public API streams to record trading data
2. **Model Training & Classification (`2_run_model.py`)**: Uses PyTorch multi-task learning to analyze wallets and classify trader behavior
3. **Visualization (`3_frontend.py`)**: Generates an interactive HTML dashboard to visualize wallet profiling results

The system monitors trading activity on Hyperliquid (a decentralized exchange on its own L1 blockchain) by subscribing to trading events and recording them in a parseable, human-readable format. The recorded data is designed for machine learning tasks related to profitable trader detection and wallet classification.

## Why WebSocket Streams vs Blockchain Events?

**WebSocket streams are recommended** for the following reasons:

1. **Lower Latency**: Real-time data delivery without waiting for block confirmations
2. **Efficiency**: No need to poll blockchain nodes or parse transaction logs
3. **Rich Data**: Direct access to trade data, user fills, and position information
4. **Reliability**: Works with free public RPC endpoints without rate limit issues
5. **Completeness**: Captures all trading activity including order book interactions

## Features

- **Real-time Trade Monitoring**: Captures all trades across all active markets on Hyperliquid
- **Comprehensive Data Collection**: Records blocks, transactions, trades, BBO, L2Book, and candles
- **ML-Ready Data Format**: JSON Lines format (one JSON object per line) for easy parsing
- **Human-Readable Output**: Timestamped, formatted log entries for manual inspection
- **Wallet Profiling**: Multi-task learning model for wallet classification
- **Market Context**: Candle data provides market conditions during trading activity

## Installation

### From PyPI (when published)

```bash
pip install hypertrack
```

### From source

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hypertrack.git
cd hypertrack
```

2. Install the package:

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1: Data Collection

Record blockchain and market data:

```bash
python -m hypertrack.1_record_data
```

This will:
- Subscribe to blockchain events (blocks, transactions)
- Subscribe to Hyperliquid API (trades, BBO, L2Book, candles)
- Save all data to `recorded_data/` directory
- Generate a data collection report

**Configuration**: Edit flags in `1_record_data.py` to enable/disable specific data types:
- `RECORD_BLOCKS = True`
- `RECORD_TRANSACTIONS = True`
- `RECORD_TRADES = True`
- `RECORD_BBO = True`
- `RECORD_L2BOOK = True`
- `RECORD_CANDLES = True`

### Stage 2: Model Training & Classification

Train the model and generate wallet classifications:

```bash
python -m hypertrack.2_run_model
```

This will:
- Load recorded data from `recorded_data/`
- Extract features from wallets
- Train a multi-task learning model
- Generate predictions and classifications
- Create `final_report.txt` with results

### Stage 3: Visualization

Generate an interactive dashboard to visualize wallet profiling results:

```bash
python -m hypertrack.3_frontend
```

This will:
- Parse `final_report.txt`
- Generate an interactive HTML dashboard (`wallet_dashboard.html`)
- Open the dashboard automatically in your browser

**Features:**
- Dark theme with modern UI
- Summary statistics (total wallets, wallets with categories)
- Grid layout showing all wallets as cards
- Interactive gauge charts for each wallet (Risk, Profitability, Bot Probability, Sophistication)
- Copy-to-clipboard functionality for wallet addresses
- Documentation modal (click "Docs" button to view README.md)

## Machine Learning Model

### Model Architecture: Multi-Task Learning (MTL)

The system uses a **PyTorch Multi-Task Learning (MTL) neural network** to simultaneously predict multiple wallet characteristics.

#### Why Multi-Task Learning?

Multi-task learning was chosen for several reasons:

1. **Shared Representations**: The model learns a shared encoder that captures common patterns across all tasks, improving generalization
2. **Data Efficiency**: By learning multiple related tasks together, the model can leverage shared information, requiring less data per task
3. **Regularization**: Learning multiple tasks acts as a form of regularization, preventing overfitting to any single task
4. **Realistic Use Case**: In practice, we want to know multiple things about a wallet simultaneously (risk, profitability, bot probability, etc.), making MTL a natural fit
5. **Transfer Learning**: Knowledge learned for one task (e.g., bot detection) can help with related tasks (e.g., sophistication scoring)

#### Model Structure

```
Input Features (37 features)
    ↓
Shared Encoder (3-layer MLP with BatchNorm & Dropout)
    ├─→ Trading Style Head (6D vector)
    ├─→ Risk Score Head (0-1)
    ├─→ Profitability Score Head (0-1)
    ├─→ Bot Probability Head (0-1)
    ├─→ Influence Score Head (≥0)
    └─→ Sophistication Score Head (0-1)
```

**Architecture Details**:
- **Input Dimension**: 37 features (see Feature Engineering section)
- **Hidden Dimensions**: 256 → 256 → 128 (shared encoder)
- **Activation**: ReLU with BatchNorm and 30% dropout for regularization
- **Output Layers**: Task-specific heads with appropriate activations (Sigmoid for probabilities, Tanh for bounded vectors, ReLU for non-negative scores)

#### Training Process

- **Optimizer**: Adam with learning rate 0.001, weight decay 1e-5
- **Loss Function**: Weighted MSE (Mean Squared Error) with task-specific weights:
  - Bot detection: 25% weight (high importance)
  - Profitability: 25% weight (high importance)
  - Risk: 15% weight
  - Sophistication: 15% weight
  - Trading Style: 10% weight
  - Influence: 10% weight
- **Epochs**: Up to 20 epochs with early stopping (patience=5)
- **Batch Size**: 32
- **Normalization**: Features are normalized using z-score normalization (mean=0, std=1)
- **Learning Rate Scheduling**: ReduceLROnPlateau (reduces LR when loss plateaus)
- **Regularization**: 
  - Output diversity regularization (prevents trivial solutions)
  - Gradient clipping (max_norm=1.0) for training stability
  - L2 weight decay
- **Early Stopping**: Stops training if no improvement for 5 epochs, loads best model

## Feature Engineering

The model uses **37 features** extracted from wallet transaction history and market context:

### Basic Transaction Features (4)
- `tx_count`: Total number of transactions
- `erc20_count`: Number of ERC-20 token transfers
- `unique_tokens`: Number of unique tokens interacted with
- `unique_addresses`: Number of unique addresses interacted with

### Temporal Features (4)
- `age_days`: Wallet age in days (first seen to last seen)
- `tx_per_day`: Average transactions per day
- `burstiness`: Coefficient of variation of inter-transaction times (measures irregularity)
- `hour_entropy`: Entropy of transaction hour distribution (measures time diversity)

**Why these metrics?**
- **Burstiness**: Detects bot-like behavior (regular intervals) vs human behavior (irregular)
- **Hour Entropy**: Low entropy = trades at specific times (possibly bot), high entropy = trades throughout day (human)

### Value Features (4)
- `total_value`: Sum of all transaction values
- `avg_value`: Average transaction value
- `max_value`: Maximum single transaction value
- `value_std`: Standard deviation of transaction values

**Why these metrics?**
- Identifies whales (high values) and consistent traders (low std)

### ERC-20 Features (2)
- `erc20_volume`: Total volume of ERC-20 transfers
- `token_diversity`: Number of unique tokens (same as `unique_tokens`)

### Directional Features (2)
- `long_ratio`: Ratio of long positions to total positions
- `flip_rate`: Rate of direction changes (long→short or short→long)

**Why these metrics?**
- **Flip Rate**: High flip rate suggests arbitrage or mean reversion strategies

### Size Features (1)
- `size_entropy`: Entropy of trade size distribution

**Why this metric?**
- Low entropy = consistent sizes (bot-like), high entropy = varied sizes (human-like)

### Network Features (2)
- `contract_interaction_count`: Number of unique contracts interacted with
- `address_diversity`: Number of unique addresses (same as `unique_addresses`)

### Market Impact Features (4)
- `avg_market_impact`: Average market impact per trade
- `max_market_impact`: Maximum market impact
- `avg_slippage`: Average slippage experienced
- `slippage_std`: Slippage consistency

### Trading Strategy Features (5)
- `momentum_score`: Tendency to follow momentum
- `mean_reversion_score`: Tendency to trade against trends
- `volatility_trading_score`: Activity during volatile periods
- `market_maker_score`: Market making activity
- `order_book_participation`: Order book interaction level

### Candle-Based Features (10) - NEW

These features provide market context during active trading periods:

- `avg_candle_volatility`: Average volatility `(high - low) / open` during active periods
- `max_candle_volatility`: Maximum volatility encountered
- `volatility_std`: Consistency of volatility levels
- `avg_candle_momentum`: Average price momentum `(close - open) / open`
- `momentum_std`: Consistency of momentum
- `positive_momentum_ratio`: Ratio of periods with positive momentum
- `avg_trend_strength`: Average trend strength `|close - open| / (high - low)`
- `avg_candle_volume`: Average volume during active periods
- `max_candle_volume`: Maximum volume encountered
- `volume_std`: Volume consistency

**Why candle features?**
- **Market Context**: Understanding market conditions (volatility, momentum, volume) when traders are active helps classify their strategies
- **Strategy Identification**: Volatility traders trade during high volatility, momentum followers trade during uptrends, contrarians trade during downtrends
- **Timing Analysis**: Reveals whether traders prefer specific market conditions

## Wallet Classification Categories

The system classifies wallets into multiple categories based on behavioral patterns. A wallet can belong to multiple categories with confidence scores.

### Bot Detection Categories

1. **Bot** (confidence: >0.7)
   - **Metrics**: `bot_probability > 0.7`
   - **Why**: ML model detects automated behavior patterns

2. **Possible Bot** (confidence: 0.5-0.7)
   - **Metrics**: `bot_probability > 0.5`
   - **Why**: Moderate likelihood of automation

### Trading Frequency Categories

3. **Scalper** (confidence: calculated)
   - **Metrics**: `tx_per_day > 50` AND `size_entropy < 0.4`
   - **Why**: High frequency + consistent sizes = scalping behavior
   - **Confidence**: `min(1.0, (tx_per_day / 100.0) * (1.0 - size_entropy))`

4. **Possible Scalper** (confidence: 0.6)
   - **Metrics**: `tx_per_day > 30` AND `size_entropy < 0.5`

5. **Active Trader** (confidence: calculated)
   - **Metrics**: `1.0 <= tx_per_day <= 50` AND `burstiness > 0.3`
   - **Why**: Moderate-high frequency with irregular timing (human-like)
   - **Confidence**: `min(1.0, tx_per_day / 50.0)`

6. **Moderate Trader** (confidence: 0.6)
   - **Metrics**: `0.5 <= tx_per_day <= 30`

7. **HODLer** (confidence: calculated)
   - **Metrics**: `tx_per_day < 0.1` AND `age_days > 30`
   - **Why**: Very low frequency + old wallet = holding strategy
   - **Confidence**: `min(1.0, (30.0 / age_days) * (0.1 / tx_per_day))`

8. **Possible HODLer** (confidence: 0.6)
   - **Metrics**: `tx_per_day < 0.5` AND `age_days > 60`

9. **Dormant** (confidence: calculated)
   - **Metrics**: `tx_count < 5` AND `age_days > 30`
   - **Why**: Very low activity despite old age

### Value-Based Categories

10. **Whale** (confidence: calculated)
    - **Metrics**: `tx_count >= 5` AND (`(max_value > 50000.0 AND total_value > 100000.0)` OR `total_value > 500000.0`)
    - **Why**: Very large transaction values indicate significant capital (whales are rare)
    - **Confidence**: `min(1.0, (max_value / 500000.0) * 0.5 + (total_value / 5000000.0) * 0.5)`
    - **Note**: Thresholds were significantly increased (100x) to avoid classifying everyone as whales

11. **Possible Whale** (confidence: calculated)
    - **Metrics**: `tx_count >= 5` AND (`(max_value > 10000.0 AND total_value > 50000.0)` OR `total_value > 200000.0`)
    - **Confidence**: `min(0.85, (max_value / 100000.0) * 0.4 + (total_value / 1000000.0) * 0.4)`

### Token Interaction Categories

12. **Token Collector** (confidence: calculated)
    - **Metrics**: `unique_tokens > 10` AND `erc20_count > tx_count * 2`
    - **Why**: High ERC-20 activity relative to regular transactions
    - **Confidence**: `min(1.0, (unique_tokens / 50.0) * (erc20_count / (tx_count * 3)))`

13. **Possible Token Collector** (confidence: 0.6)
    - **Metrics**: `unique_tokens > 5` AND `erc20_count > tx_count`

14. **Arbitrageur** (confidence: calculated)
    - **Metrics**: `unique_tokens > 5` AND `flip_rate > 0.5`
    - **Why**: Multiple tokens + high direction changes = arbitrage
    - **Confidence**: `min(1.0, (unique_tokens / 20.0) * flip_rate)`

15. **Possible Arbitrageur** (confidence: 0.6)
    - **Metrics**: `unique_tokens > 3` AND `flip_rate > 0.3`

### Market Condition Categories (Candle-Based) - NEW

16. **Volatility Trader** (confidence: calculated)
    - **Metrics**: `avg_candle_volatility > 0.02` (2%+) AND `tx_count > 5`
    - **Why**: Trades during high volatility periods (volatility trading strategy)
    - **Confidence**: `min(1.0, (avg_volatility / 0.05) * (tx_count / 20.0))`

17. **Possible Volatility Trader** (confidence: 0.6)
    - **Metrics**: `avg_candle_volatility > 0.01` (1%+)

18. **Momentum Follower** (confidence: calculated)
    - **Metrics**: `positive_momentum_ratio > 0.7` AND `avg_momentum > 0.001` AND `tx_count > 5`
    - **Why**: Trades primarily when price is rising (momentum strategy)
    - **Confidence**: `min(1.0, positive_momentum_ratio * (avg_momentum / 0.01))`

19. **Possible Momentum Follower** (confidence: 0.6)
    - **Metrics**: `positive_momentum_ratio > 0.6`

20. **Contrarian Trader** (confidence: calculated)
    - **Metrics**: `positive_momentum_ratio < 0.3` AND `avg_momentum < -0.001` AND `tx_count > 5`
    - **Why**: Trades when price is falling (mean reversion/contrarian strategy)
    - **Confidence**: `min(1.0, (1.0 - positive_momentum_ratio) * (abs(avg_momentum) / 0.01))`

21. **Possible Contrarian Trader** (confidence: 0.6)
    - **Metrics**: `positive_momentum_ratio < 0.4`

22. **High Volume Trader** (confidence: calculated)
    - **Metrics**: `avg_candle_volume > 100.0` AND `tx_count > 5`
    - **Why**: Prefers trading during high volume periods (liquidity seeking)
    - **Confidence**: `min(1.0, (avg_volume / 500.0) * (tx_count / 20.0))`

### ML-Driven Categories (NEW)

These categories are based primarily on ML model predictions, enhanced with rule-based validation:

23. **Profitable Trader** (confidence: calculated)
    - **Metrics**: `profitability_score > 0.6` AND `tx_count >= 3`
    - **Why**: ML model identifies wallets with high profitability patterns
    - **Confidence**: `min(1.0, profitability_score * 0.7 + sophistication_score * 0.3)`

24. **Possibly Profitable Trader** (confidence: calculated)
    - **Metrics**: `profitability_score > 0.4` AND `tx_count >= 3`
    - **Confidence**: `min(0.85, profitability_score * 0.6 + sophistication_score * 0.2)`

25. **High Risk Trader** (confidence: calculated)
    - **Metrics**: `risk_score > 0.6` AND `tx_count >= 3`
    - **Why**: ML model identifies wallets with high risk-taking behavior
    - **Confidence**: `min(1.0, risk_score * 0.7 + sophistication_score * 0.2)`

26. **Moderate Risk Trader** (confidence: calculated)
    - **Metrics**: `risk_score > 0.4` AND `tx_count >= 3`
    - **Confidence**: `min(0.85, risk_score * 0.6)`

27. **Sophisticated Trader** (confidence: calculated)
    - **Metrics**: `sophistication_score > 0.6` AND `tx_count >= 5`
    - **Why**: ML model identifies wallets with sophisticated trading patterns
    - **Confidence**: `min(1.0, sophistication_score * 0.6 + profitability_score * 0.3 + risk_score * 0.1)`

28. **Possibly Sophisticated Trader** (confidence: calculated)
    - **Metrics**: `sophistication_score > 0.4` AND `tx_count >= 3`
    - **Confidence**: `min(0.85, sophistication_score * 0.7)`

29. **Influential Trader** (confidence: calculated)
    - **Metrics**: `influence_score > 3.0` AND `tx_count >= 5`
    - **Why**: ML model identifies wallets with high market influence (market makers, large traders)
    - **Confidence**: `min(1.0, min(1.0, influence_score / 50.0) * 0.7 + sophistication_score * 0.3)`

30. **Possibly Influential Trader** (confidence: calculated)
    - **Metrics**: `influence_score > 1.0` AND `tx_count >= 3`
    - **Confidence**: `min(0.85, min(1.0, influence_score / 20.0) * 0.6)`

### Classification Logic

Categories are determined using a **hybrid approach** combining rule-based classification with ML-enhanced confidence:

1. **Rule-Based Foundation**: Each category has specific thresholds for relevant metrics (transaction frequency, values, patterns)
2. **ML Enhancement**: ML model scores (risk, profitability, sophistication, bot probability, trading style vector) are used to:
   - Boost confidence when ML confirms the category
   - Add new ML-driven categories (Profitable Trader, Sophisticated Trader, etc.)
   - Refine category confidence based on learned patterns
3. **Multi-Category Assignment**: A wallet can belong to multiple categories (e.g., "Bot" + "Scalper" + "Profitable Trader")
4. **Ranking**: Categories are sorted by confidence score (highest first)

**Hybrid Approach Benefits:**
- **Interpretability**: Rule-based thresholds are transparent and explainable
- **Domain Knowledge**: Incorporates trading domain expertise through rules
- **ML Intelligence**: ML model learns patterns that rules might miss
- **Robustness**: Rule-based categories work even when ML scores are low (ML is optional boost)
- **Flexibility**: Easy to adjust thresholds based on observed data

**How ML Enhances Categories:**
- **Bot Classification**: Uses `bot_probability` directly, boosted by `sophistication_score`
- **Scalper**: Rule-based frequency/size + ML sophistication/risk boost
- **Active Trader**: Rule-based frequency + ML profitability/sophistication boost
- **Arbitrageur**: Rule-based tokens/flip_rate + ML profitability/sophistication boost
- **Strategy Categories**: Rule-based market conditions + Trading Style vector + ML profitability
- **New ML Categories**: Purely ML-driven (Profitable, Risk, Sophisticated, Influential)

### Why Some Wallets Have No Categories

A wallet may show **"Categories: None"** in the report. This does **not** mean the model is inconclusive or that the wallet wasn't analyzed. It means the wallet doesn't match any of the predefined category thresholds.

**Reasons for No Categories:**

1. **Strict Thresholds**: Category thresholds are intentionally strict to identify clear behavioral patterns:
   - **Bot**: Requires `bot_probability > 0.7` (very high confidence)
   - **Scalper**: Requires `tx_per_day > 50` (very high frequency)
   - **Whale**: Requires `max_value > 50000.0` AND `total_value > 100000.0` OR `total_value > 500000.0` (very large transactions - thresholds were increased 100x)
   - **Volatility Trader**: Requires `avg_candle_volatility > 0.02` (2%+ volatility) AND `tx_count > 5`

2. **Normal/Low-Activity Wallets**: Many wallets are simply normal traders with:
   - Moderate transaction frequency (not high enough for Scalper, not low enough for HODLer)
   - Moderate transaction values (not large enough for Whale)
   - Mixed trading patterns (don't clearly fit any strategy category)
   - Low activity (insufficient data to classify)

3. **Insufficient Data**: Wallets with very few transactions may not meet category requirements:
   - Most categories require `tx_count > 5` or specific frequency thresholds
   - New wallets or wallets with limited activity won't have enough data

4. **Missing Market Context**: If candle data is unavailable or doesn't match transaction timestamps:
   - Candle-based categories (Volatility Trader, Momentum Follower, etc.) won't match
   - This is normal if the wallet traded when no candle data was recorded

**What This Means:**

- **The ML model still works**: All wallets receive ML scores (bot probability, risk, profitability, etc.)
- **Categories are optional labels**: They provide behavioral context but aren't required for analysis
- **"None" is informative**: It indicates the wallet doesn't fit any extreme behavioral pattern
- **You can adjust thresholds**: Modify `classify_wallet()` in `2_run_model.py` to lower thresholds or add more general categories

**Configuration:**

- Set `SHOW_NO_CATEGORIES = False` in `2_run_model.py` to exclude wallets with no categories from the report
- Set `TX_COUNT_THRESHOLD` to filter out wallets with too few transactions

## Output Format

### Data Collection Output

Data is saved in **JSON Lines format** (JSONL) in the `recorded_data/` directory:

- `blocks.jsonl`: Blockchain blocks
- `transactions.jsonl`: All transactions
- `trades.jsonl`: Trade data from Hyperliquid API
- `bbo.jsonl`: Best Bid Offer updates
- `l2book.jsonl`: Level 2 order book updates
- `candles.jsonl`: Completed OHLCV candles (1-minute intervals)
- `data_collection_report.txt`: Summary of collected data

### Model Output

- `wallet_model.pt`: Trained PyTorch model checkpoint
- `final_report.txt`: Human-readable report with wallet classifications

#### Example Report

```
================================================================================
WALLET PROFILING REPORT
================================================================================
Generated: 2026-01-22T12:00:00Z
Total wallets analyzed: 150

================================================================================

Wallet #1: 0x1234...
--------------------------------------------------------------------------------
Categories: Bot (0.85), Scalper (0.72), Momentum Follower (0.68)

Trading Style Score (6D vector): ['0.2341', '-0.1234', '0.5678', '0.1234', '-0.2341', '0.3456']
Risk Score: 0.7234
Profitability Score: 0.8123
Bot Probability: 0.8500
Influence Score: 5.2341
Sophistication Score: 0.6789

Transaction Count: 1250
Age (days): 45.23
```

## Challenges and Solutions in Wallet Categorization

### Initial Challenges

During development, we encountered several significant hurdles in creating an effective wallet categorization system:

#### 1. **Everyone Classified as Whales**

**Problem**: Initially, almost all wallets were being classified as "Whale" or "Possible Whale", making the categorization useless.

**Root Cause**: The original thresholds were too low:
- Whale: `max_value > 100.0` OR `total_value > 1000.0`
- Possible Whale: `max_value > 10.0` OR `total_value > 100.0`

In crypto markets, especially on Hyperliquid, these values are quite common, so most participants met the criteria.

**Solution**: Increased thresholds by 100-500x:
- Whale: `max_value > 50000.0` AND `total_value > 100000.0` OR `total_value > 500000.0`
- Possible Whale: `max_value > 10000.0` AND `total_value > 50000.0` OR `total_value > 200000.0`
- Added requirement: `tx_count >= 5` to avoid one-off large transactions

This made whale classification much more exclusive and meaningful.

#### 2. **ML Model Not Contributing to Categorization**

**Problem**: The ML model was trained and provided scores, but categories were purely rule-based. The ML scores weren't being used effectively, making the ML component feel disconnected.

**Root Cause**: 
- Categories were determined by simple threshold rules on raw features
- Only `bot_probability` was used (for Bot classification)
- Other ML scores (risk, profitability, sophistication, trading style) were reported but not used for categorization

**Solution**: Implemented hybrid ML-enhanced categorization:
- **ML scores integrated**: All ML scores now influence category confidence
- **ML-driven categories added**: New categories based purely on ML predictions (Profitable Trader, Sophisticated Trader, etc.)
- **Trading Style vector used**: 6D style vector helps identify strategy preferences (momentum, mean reversion, volatility)
- **ML as boost, not requirement**: Rule-based categories still work even with low ML confidence; ML provides enhancement when available

#### 3. **No Categories Assigned (0 Wallets with Categories)**

**Problem**: After implementing ML enhancements, sometimes no wallets were getting categories assigned.

**Root Cause**: 
- ML confidence filter was too strict: removed categories when `ml_confidence < 0.2`
- ML boost thresholds were too high: required `ml_confidence > 0.3` to apply boosts
- ML-driven category thresholds were too strict (0.7/0.5 for scores, high tx_count requirements)

**Solution**:
- **Removed strict ML filter**: Rule-based categories now work regardless of ML confidence
- **Lowered ML boost threshold**: Changed from `ml_confidence > 0.3` to `ml_confidence > 0.1`
- **Lowered ML category thresholds**: Reduced from 0.7/0.5 to 0.6/0.4, tx_count from 5/10 to 3/5
- **ML is optional**: Categories work based on rules; ML provides boost when available

#### 4. **Self-Supervised Learning Limitations**

**Problem**: The ML model uses self-supervised learning (training against zeros), which may not learn meaningful patterns.

**Solution**: Enhanced training process:
- **Weighted loss**: Higher weight for important tasks (bot, profitability)
- **Output diversity regularization**: Prevents model from collapsing to trivial solutions
- **Better training**: Increased epochs (10→20), learning rate scheduling, early stopping
- **Gradient clipping**: Improves training stability

**Note**: For production use, consider supervised learning with labeled data if available.

### Current Approach

The system now uses a **hybrid rule-based + ML-enhanced** approach:

1. **Rule-based foundation**: Categories are primarily determined by interpretable rules on features
2. **ML enhancement**: ML scores boost confidence and add new categories
3. **Robustness**: Works even when ML scores are low (ML is enhancement, not requirement)
4. **Flexibility**: Easy to adjust thresholds and add new categories

This approach balances interpretability (rules) with intelligence (ML), ensuring the system works reliably while learning from data.

## Three-Stage Architecture

### Stage 1: Data Collection (`1_record_data.py`)

**Purpose**: Collect and store raw data for later analysis

**Features**:
- Runs continuously until stopped (Ctrl+C)
- Clears `recorded_data/` folder on startup
- Optional recording flags for each data type
- Real-time status updates
- Automatic reconnection on connection loss
- Filters intra-candle updates (only completed candles)

**Why Separate Stage?**
- Allows data collection to run independently
- Enables multiple model runs on same dataset
- Separates data collection from analysis concerns

### Stage 2: Model Training & Classification (`2_run_model.py`)

**Purpose**: Analyze collected data and generate wallet classifications

**Features**:
- Loads all recorded data
- Extracts features from wallets
- Trains multi-task learning model
- Generates predictions and classifications
- Creates comprehensive report (`final_report.txt`)

**Why Separate Stage?**
- Can run offline on collected data
- Allows experimentation with different models
- Enables batch processing of historical data

### Stage 3: Visualization (`3_frontend.py`)

**Purpose**: Generate interactive dashboard to visualize wallet profiling results

**Features**:
- Parses `final_report.txt` to extract wallet data
- Creates interactive HTML dashboard with dark theme
- Displays wallet cards in responsive grid layout
- Shows summary statistics (total wallets, wallets with categories)
- Interactive gauge charts for each wallet (Risk, Profitability, Bot Probability, Sophistication)
- Copy-to-clipboard functionality for wallet addresses
- Documentation modal (displays README.md content)
- Opens automatically in browser

**Why Separate Stage?**
- Can be run independently after model training
- Generates static HTML file (works offline)
- Provides visual interface for exploring results
- Separates visualization from data collection and analysis

## API Endpoints

The package uses Hyperliquid's official public API endpoints:

- **Mainnet**: `https://api.hyperliquid.xyz` (default)
- **WebSocket**: `wss://api.hyperliquid.xyz/ws` (for real-time streams)
- **Blockchain RPC**: Alchemy RPC endpoint (configured in code)

These are free, public, and reliable endpoints provided by Hyperliquid.

## Requirements

- Python 3.8+
- `torch` (PyTorch for ML model)
- `numpy` (numerical operations)
- `requests` (for HTTP REST API calls)
- `websocket-client` (for WebSocket real-time streams)

## File Structure

```
hypertrack/
├── __init__.py              # Package initialization
├── 1_record_data.py         # Stage 1: Data collection
├── 2_run_model.py          # Stage 2: Model training & classification
├── 3_frontend.py            # Stage 3: Interactive dashboard visualization
├── recorder_trades.py       # Example: Trade recorder for testing Hyperliquid API
├── recorder_bbo.py          # Example: BBO recorder for testing Hyperliquid API
├── recorder_l2Book.py       # Example: L2Book recorder for testing Hyperliquid API
└── recorder_candle.py       # Example: Candle recorder for testing Hyperliquid API

**Note**: The `recorder_*.py` files are example separate Hyperliquid feed recorders used to test the Hyperliquid API. They demonstrate how to subscribe to individual data streams (trades, BBO, L2Book, candles) and can be used independently for testing purposes. The main data collection is handled by `1_record_data.py`, which integrates all these streams.

recorded_data/               # Data collection output
├── blocks.jsonl
├── transactions.jsonl
├── trades.jsonl
├── bbo.jsonl
├── l2book.jsonl
├── candles.jsonl
└── data_collection_report.txt

sample_data/                 # Sample recorder outputs
├── trades_ETH.log
├── bbo_ETH.log
├── l2Book_ETH.log
└── candle_ETH.log
```

## Important Notes

1. **No Private Data**: Does NOT subscribe to `userEvents` or `userTrades` (those are private)
2. **Public Blockchain Only**: Only uses publicly available blockchain data
3. **Candle Filtering**: Only completed candles are recorded (intra-candle updates filtered)
4. **Data Clearing**: `recorded_data/` folder is cleared on each run of `1_record_data.py`
5. **GPU Support**: Model automatically uses GPU if available (CUDA)

## Limitations

- **Rate Limits**: The recorder includes delays to respect API rate limits
- **Historical Data**: The recorder starts from the current time. For historical data, use Hyperliquid's historical API endpoints separately
- **Model Training**: Requires sufficient data (50+ wallets recommended for meaningful results)
- **Candle Matching**: Candles are matched to transactions by timestamp; if no candle exists for a transaction period, candle features default to 0

## Future Enhancements

- [ ] Real-time streaming classification (classify wallets as data arrives)
- [ ] Database storage option (SQLite, PostgreSQL)
- [ ] Integration with live trading signals
- [ ] Additional market data sources
- [ ] Model versioning and A/B testing
- [ ] Web dashboard for visualization

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This tool is for monitoring and analysis purposes only. Always do your own research and never invest more than you can afford to lose. Trading cryptocurrencies involves substantial risk.
