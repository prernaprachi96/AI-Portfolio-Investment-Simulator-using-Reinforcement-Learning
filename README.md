# 🤖 AI Portfolio Investment Simulator using Reinforcement Learning

A Deep Q-Network (DQN) based trading agent that learns to make **Buy, Sell, and Hold** decisions on real stock market data.

---

## 📁 Project Structure

```
portfolio_rl/
├── main.py           ← Run this to train + backtest
├── environment.py    ← Custom Gym trading environment
├── agent.py          ← DQN agent with experience replay
├── trainer.py        ← Training loop + backtesting + plots
├── data_loader.py    ← Download real stock data (yfinance)
├── requirements.txt  ← Dependencies
└── outputs/          ← Results saved here (auto-created)
```

---

## ⚙️ Setup

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the simulator
python main.py
```

---

## 🧠 How It Works

### Environment (`environment.py`)
- **State**: A sliding window of 20 days × 6 features:
  - Normalized Close Price
  - MA5 (5-day moving average)
  - MA20 (20-day moving average)
  - Volatility (10-day rolling std of returns)
  - RSI (Relative Strength Index)
  - MACD signal
- **Actions**: `0=Hold`, `1=Buy`, `2=Sell`
- **Reward**: Sharpe-inspired risk-adjusted reward — rewards returns, penalizes volatility and drawdown

### Agent (`agent.py`)
- **DQN** with 3-layer neural network
- **Experience Replay Buffer** (50,000 transitions)
- **Target Network** for stable training (updated every 10 steps)
- **Epsilon-greedy** exploration (decays from 1.0 → 0.01)

### Training (`main.py`)
- Downloads real data via `yfinance`
- Splits: 70% train / 15% validation / 15% test
- Trains for 150 episodes on training data
- Backtests on unseen test data
- Compares vs **Buy-and-Hold** benchmark

---

## 📊 Output

For each stock, you get:
- **Portfolio value chart** — RL Agent vs Buy & Hold
- **Performance metrics** — Total Return, Sharpe Ratio, Max Drawdown
- **Training curves** — Reward, portfolio value, loss over episodes
- **Action pie chart** — How often the agent buys/sells/holds
- **Drawdown chart** — Risk visualization
- **Saved model** — `.pth` file for reuse

---

## ⚙️ Configuration (in `main.py`)

```python
CONFIG = {
    "tickers":          ["AAPL", "MSFT", "TSLA"],   # Stocks to trade
    "start_date":       "2018-01-01",
    "end_date":         "2024-01-01",
    "num_episodes":     150,                          # Training episodes
    "initial_balance":  10_000,                       # Starting capital ($)
    "window_size":      20,                           # Lookback window (days)
    "transaction_cost": 0.001,                        # 0.1% per trade
    "learning_rate":    1e-3,
    "gamma":            0.99,
    "epsilon_decay":    0.995,
    "batch_size":       64,
}
```

---

## 📈 Metrics Explained

| Metric | Description |
|---|---|
| Total Return | % gain/loss vs initial investment |
| Sharpe Ratio | Return per unit of risk (annualized) |
| Max Drawdown | Largest peak-to-trough loss |
| # Trades | Number of buy/sell actions taken |

---

## ⚠️ Disclaimer
This is an academic/research project. Do not use for real trading decisions.
