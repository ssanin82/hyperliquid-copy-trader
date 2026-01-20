# hypertrack

A Python package for tracking Hyperliquid trading activity and detecting profitable traders using real-time data streams.

## Overview

`hypertrack` monitors trading activity on Hyperliquid (a decentralized exchange on its own L1 blockchain) by subscribing to trading events and recording them in a parseable, human-readable format. The recorded data is designed for machine learning tasks related to profitable trader detection.

## Why WebSocket Streams vs Blockchain Events?

**WebSocket streams are recommended** for the following reasons:

1. **Lower Latency**: Real-time data delivery without waiting for block confirmations
2. **Efficiency**: No need to poll blockchain nodes or parse transaction logs
3. **Rich Data**: Direct access to trade data, user fills, and position information
4. **Reliability**: Works with free public RPC endpoints without rate limit issues
5. **Completeness**: Captures all trading activity including order book interactions

## Features

- **Real-time Trade Monitoring**: Captures all trades across all active markets on Hyperliquid
- **User Fill Tracking**: Monitors fills for specific addresses to track profitable traders
- **ML-Ready Data Format**: JSON Lines format (one JSON object per line) for easy parsing
- **Human-Readable Output**: Timestamped, formatted log entries for manual inspection
- **Comprehensive Event Data**: Includes price, size, side, user addresses, PnL, and more
- **Reliable Public RPC**: Uses Hyperliquid's official public API endpoints (free and reliable)

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

### Basic Usage

After installation, you can use the command-line interface:

```bash
hypertrack
```

Or run as a module:

```bash
python -m hypertrack.recorder
```

This will create a file called `hyperliquid_trades.log` in the current directory.

### Custom Output File

Specify a custom output file:

```bash
python -m hypertrack.recorder -f my_trades.log
```

### Programmatic Usage

```python
from hypertrack import TradeRecorder

# Create a recorder instance
recorder = TradeRecorder(filename="trades.log")

# Start recording (runs until Ctrl+C)
recorder.start()
```

## Output Format

The recorder writes events in **JSON Lines format** (JSONL), where each line is a complete JSON object. This format is:

- **Human-readable**: Easy to inspect with `cat`, `head`, `tail`, or any text editor
- **Parseable**: Each line can be parsed independently as JSON
- **ML-friendly**: Perfect for streaming data processing and machine learning pipelines

### Example Output

```json
{"type":"header","timestamp":"2024-01-15T10:30:00.000Z","recorder_version":"0.1.0","description":"Hyperliquid trading activity log - JSON Lines format"}
{"timestamp":"2024-01-15T10:30:05.123Z","unix_timestamp":1705315805.123,"event_type":"trade","coin":"BTC","side":"B","price":42000.5,"size":0.1,"notional_usd":4200.05,"data":{"coin":"BTC","side":"B","px":"42000.5","sz":"0.1","time":1705315805123,"hash":"0x..."}}
{"timestamp":"2024-01-15T10:30:06.456Z","unix_timestamp":1705315806.456,"event_type":"user_fill","coin":"ETH","side":"A","price":2500.75,"size":2.5,"notional_usd":6251.875,"user_address":"0x1234...","closed_pnl":125.50,"data":{...}}
```

### Event Fields

Each trade event includes:

- `timestamp`: ISO 8601 formatted UTC timestamp
- `unix_timestamp`: Unix timestamp for easy time-based filtering
- `event_type`: Type of event (`trade`, `user_fill`, etc.)
- `coin`: Trading pair symbol (e.g., "BTC", "ETH")
- `side`: Trade side (`"B"` for buy/bid, `"A"` for sell/ask)
- `price`: Execution price
- `size`: Trade size
- `notional_usd`: Total trade value in USD (price × size)
- `user_address`: Wallet address (for user fills)
- `closed_pnl`: Realized profit/loss (when available)
- `data`: Complete raw event data from Hyperliquid API

## Events Tracked

The recorder captures the following events, which provide sufficient information for ML-based profitable trader detection:

### 1. **Trades** (`event_type: "trade"`)
- All executed trades across all markets
- Price, size, side, timestamp
- Trade hash for deduplication
- Coin symbol

### 2. **User Fills** (`event_type: "user_fill"`)
- Fills for specific user addresses
- Entry/exit prices
- Position size
- Realized PnL (profit/loss)
- Liquidation prices
- Order IDs

### 3. **Position Updates** (future)
- Open positions
- Unrealized PnL
- Leverage information
- Margin ratios

## ML Features for Profitable Trader Detection

The recorded data enables the following ML analysis:

1. **Win Rate Calculation**: Track user fills to calculate win/loss ratios
2. **Profitability Metrics**: Use `closed_pnl` to identify consistently profitable traders
3. **Trading Patterns**: Analyze timing, frequency, and size of trades
4. **Market Impact**: Compare trade size to order book depth
5. **Risk Assessment**: Monitor liquidation prices and leverage
6. **Time-Series Analysis**: Track performance over time using timestamps

## API Endpoints

The package uses Hyperliquid's official public API endpoints:

- **Mainnet**: `https://api.hyperliquid.xyz` (default)
- **WebSocket**: `wss://api.hyperliquid.xyz/ws` (for real-time streams)

These are free, public, and reliable endpoints provided by Hyperliquid.

## Stopping the Recorder

Press `Ctrl+C` to gracefully stop the recorder. The recorder will:

1. Write a footer with statistics
2. Close the log file properly
3. Display summary statistics

## File Structure

```
hypertrack/
├── __init__.py          # Package initialization
├── recorder.py          # Main recording module
├── stream_client.py     # WebSocket client (for future use)
└── requirements.txt     # Python dependencies
```

## Requirements

- Python 3.8+
- `hyperliquid-python-sdk` (official Hyperliquid Python SDK)
- `websocket-client` (optional, for WebSocket streams)

## Limitations

- **Rate Limits**: The recorder includes delays to respect API rate limits. Adjust polling intervals in the code if needed.
- **User Address Discovery**: Currently tracks users discovered from recent trades. For comprehensive tracking, you may need to provide a list of addresses to monitor.
- **Historical Data**: The recorder starts from the current time. For historical data, use Hyperliquid's historical API endpoints separately.

## Future Enhancements

- [ ] Direct WebSocket stream support for lower latency
- [ ] Configurable event filters (by coin, user, size, etc.)
- [ ] Real-time profitability scoring
- [ ] Database storage option (SQLite, PostgreSQL)
- [ ] Integration with ML models for live trader ranking

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This tool is for monitoring and analysis purposes only. Always do your own research and never invest more than you can afford to lose. Trading cryptocurrencies involves substantial risk.
