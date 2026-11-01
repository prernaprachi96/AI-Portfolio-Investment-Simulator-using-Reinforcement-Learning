"""
============================================================
  AI Portfolio Investment Simulator using Reinforcement Learning
  DQN Agent for Stock Trading
  Actions: Buy | Sell | Hold
  Reward:  Risk-adjusted (Sharpe-inspired)
  Benchmark: Buy-and-Hold comparison
============================================================

SETUP:
    pip install gymnasium torch yfinance pandas numpy matplotlib

RUN:
    python portfolio_rl_simulator.py
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

warnings.filterwarnings("ignore")

# ─── Try importing required libraries ────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("❌ PyTorch not found. Run: pip install torch")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("❌ yfinance not found. Run: pip install yfinance")
    sys.exit(1)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("❌ gymnasium not found. Run: pip install gymnasium")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  Edit these values to customize your experiment
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "tickers":           ["AAPL", "MSFT", "TSLA"],   # Stocks to trade
    "start_date":        "2018-01-01",
    "end_date":          "2024-01-01",
    "num_episodes":      150,
    "initial_balance":   10_000,
    "window_size":       20,       # Lookback window in days
    "transaction_cost":  0.001,    # 0.1% per trade
    "learning_rate":     1e-3,
    "gamma":             0.99,
    "epsilon":           1.0,
    "epsilon_min":       0.01,
    "epsilon_decay":     0.995,
    "batch_size":        64,
    "target_update":     10,
    "save_dir":          "outputs",
}


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADER
# ═══════════════════════════════════════════════════════════════════

def download_stock_data(tickers, start_date, end_date):
    """Download historical OHLCV stock data using yfinance."""
    print(f"\n📥 Downloading data for: {tickers}")
    stock_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 100:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.dropna(inplace=True)
                stock_data[ticker] = df
                print(f"  ✅ {ticker}: {len(df)} trading days")
            else:
                print(f"  ⚠️  {ticker}: Not enough data, skipping")
        except Exception as e:
            print(f"  ❌ {ticker}: Failed — {e}")
    return stock_data


def split_data(df, train_ratio=0.70, val_ratio=0.15):
    """Split into train / validation / test sets."""
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train = df.iloc[:t1].reset_index(drop=True)
    val   = df.iloc[t1:t2].reset_index(drop=True)
    test  = df.iloc[t2:].reset_index(drop=True)
    print(f"\n📊 Data Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)} days")
    return train, val, test


def compute_bh_stats(df, initial_balance=10_000):
    """Buy at day 0, hold until end — full stats."""
    prices = df['Close'].values
    shares = initial_balance / prices[0]
    history = [shares * p for p in prices]
    values  = np.array(history)
    returns = np.diff(values) / (values[:-1] + 1e-8)
    final   = history[-1]
    peak    = np.maximum.accumulate(values)
    mdd     = np.max((peak - values) / (peak + 1e-8)) * 100
    sharpe  = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
    return {
        "label": "Buy & Hold",
        "final_value": final,
        "total_return_pct": (final - initial_balance) / initial_balance * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": mdd,
        "portfolio_history": history,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: TRADING ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

class PortfolioTradingEnv(gym.Env):
    """
    Custom Gym environment for single-stock RL trading.

    State  : window_size × 6 features + 2 portfolio state values
    Actions: 0 = Hold | 1 = Buy | 2 = Sell
    Reward : Sharpe-inspired risk-adjusted return
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance=10_000, window_size=20,
                 transaction_cost=0.001, reward_scaling=100.0):
        super().__init__()
        self.df               = df.reset_index(drop=True)
        self.initial_balance  = initial_balance
        self.window_size      = window_size
        self.transaction_cost = transaction_cost
        self.reward_scaling   = reward_scaling
        self.n_features       = 6   # close_norm, ma5, ma20, volatility, rsi, macd

        self.action_space      = spaces.Discrete(3)
        obs_dim                = self.window_size * self.n_features + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._precompute_features()

    def _precompute_features(self):
        df = self.df.copy()

        # Normalized close
        df['close_norm'] = df['Close'] / df['Close'].iloc[0]

        # Moving averages (ratio to current close)
        df['ma5']  = df['Close'].rolling(5).mean()  / df['Close']
        df['ma20'] = df['Close'].rolling(20).mean() / df['Close']

        # Volatility: rolling std of daily returns
        df['returns']    = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()

        # RSI (normalized to [0,1])
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-8)
        df['rsi'] = 1 - (1 / (1 + rs))

        # MACD signal
        ema12       = df['Close'].ewm(span=12, adjust=False).mean()
        ema26       = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd']  = (ema12 - ema26) / df['Close']

        df.fillna(0, inplace=True)
        self.features = df[['close_norm','ma5','ma20','volatility','rsi','macd']].values
        self.prices   = df['Close'].values
        self.returns_series = df['returns'].fillna(0).values

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step      = self.window_size
        self.balance           = self.initial_balance
        self.shares_held       = 0
        self.portfolio_value   = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        self.period_returns    = []
        self.trades            = []
        return self._get_obs(), {}

    def _get_obs(self):
        window      = self.features[self.current_step - self.window_size : self.current_step]
        flat        = window.flatten()
        position    = 1.0 if self.shares_held > 0 else 0.0
        bal_ratio   = self.balance / self.initial_balance
        return np.concatenate([flat, [position, bal_ratio]]).astype(np.float32)

    def step(self, action):
        price    = self.prices[self.current_step]
        prev_val = self.portfolio_value

        if action == 1:   # BUY
            if self.balance > price:
                qty = int(self.balance * 0.95 / price)
                if qty > 0:
                    self.balance     -= qty * price * (1 + self.transaction_cost)
                    self.shares_held += qty
                    self.trades.append(('BUY', self.current_step, price, qty))

        elif action == 2:  # SELL
            if self.shares_held > 0:
                self.balance += self.shares_held * price * (1 - self.transaction_cost)
                self.trades.append(('SELL', self.current_step, price, self.shares_held))
                self.shares_held = 0

        self.portfolio_value = self.balance + self.shares_held * price
        self.portfolio_history.append(self.portfolio_value)

        ret = (self.portfolio_value - prev_val) / (prev_val + 1e-8)
        self.period_returns.append(ret)

        reward    = self._compute_reward(ret)
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        obs = self._get_obs() if not terminated else \
              np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, False, {"portfolio_value": self.portfolio_value}

    def _compute_reward(self, period_return):
        if len(self.period_returns) < 5:
            return period_return * self.reward_scaling

        recent  = np.array(self.period_returns[-20:])
        sharpe  = np.mean(recent) / (np.std(recent) + 1e-8)

        peak     = max(self.portfolio_history)
        drawdown = (peak - self.portfolio_value) / (peak + 1e-8)

        return float((sharpe - 0.5 * drawdown) * self.reward_scaling)

    def get_stats(self):
        vals    = np.array(self.portfolio_history)
        rets    = np.diff(vals) / (vals[:-1] + 1e-8)
        peak    = np.maximum.accumulate(vals)
        mdd     = np.max((peak - vals) / (peak + 1e-8)) * 100
        sharpe  = (np.mean(rets) / (np.std(rets) + 1e-8)) * np.sqrt(252)
        return {
            "final_value":      self.portfolio_value,
            "total_return_pct": (self.portfolio_value - self.initial_balance)
                                 / self.initial_balance * 100,
            "sharpe_ratio":     sharpe,
            "max_drawdown_pct": mdd,
            "num_trades":       len(self.trades),
            "portfolio_history": self.portfolio_history,
            "trades":           self.trades,
        }

    def render(self, mode="human"):
        print(f"Step {self.current_step} | "
              f"Portfolio: ${self.portfolio_value:.2f} | "
              f"Balance: ${self.balance:.2f} | Shares: {self.shares_held}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: DQN AGENT
# ═══════════════════════════════════════════════════════════════════

class DQNNetwork(nn.Module):
    """3-layer neural network: state → Q-values for [Hold, Buy, Sell]."""
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size experience replay buffer."""
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buf.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a),
                np.array(r,  dtype=np.float32),
                np.array(s_, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """
    Deep Q-Network agent with:
      - Epsilon-greedy exploration
      - Experience replay
      - Target network (updated every N steps)
    """
    def __init__(self, state_dim, action_dim=3, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=10):

        self.action_dim        = action_dim
        self.gamma             = gamma
        self.epsilon           = epsilon
        self.epsilon_min       = epsilon_min
        self.epsilon_decay     = epsilon_decay
        self.batch_size        = batch_size
        self.target_update_freq = target_update_freq
        self.train_step        = 0
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net      = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer    = ReplayBuffer()

        n_params = sum(p.numel() for p in self.q_net.parameters())
        print(f"🤖 DQN Agent | Device: {self.device} | "
              f"State: {state_dim} | Params: {n_params:,}")

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(t).argmax().item()

    def store(self, s, a, r, s_, done):
        self.buffer.push(s, a, r, s_, done)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s_, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        curr_q   = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(s_).max(1)[0]
            tgt_q  = r + self.gamma * next_q * (1 - d)

        loss = self.criterion(curr_q, tgt_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({"q_net": self.q_net.state_dict(), "epsilon": self.epsilon}, path)
        print(f"💾 Model saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.epsilon = ckpt["epsilon"]


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: TRAINING & BACKTESTING
# ═══════════════════════════════════════════════════════════════════

def train_agent(agent, env, num_episodes=100):
    """Run training loop for given number of episodes."""
    print(f"\n🏋️  Training for {num_episodes} episodes...")
    print("─" * 60)

    ep_rewards, ep_values, ep_losses = [], [], []

    for ep in range(1, num_episodes + 1):
        state, _     = env.reset()
        total_reward = 0
        losses       = []

        while True:
            action              = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done                = terminated or truncated
            agent.store(state, action, reward, next_state, float(done))
            loss                = agent.train()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            state         = next_state
            if done:
                break

        stats = env.get_stats()
        ep_rewards.append(total_reward)
        ep_values.append(stats['final_value'])
        if losses:
            ep_losses.append(np.mean(losses))

        if ep % 10 == 0 or ep == 1:
            avg_r = np.mean(ep_rewards[-10:])
            avg_v = np.mean(ep_values[-10:])
            print(f"  Ep {ep:3d}/{num_episodes} | "
                  f"Avg Reward: {avg_r:8.2f} | "
                  f"Avg Portfolio: ${avg_v:9.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Return: {stats['total_return_pct']:+.1f}%")

    print("✅ Training complete!")
    return ep_rewards, ep_values, ep_losses


def run_backtest(agent, env):
    """Run a full episode with no exploration (pure exploitation)."""
    state, _    = env.reset()
    action_log  = []
    while True:
        action = agent.select_action(state, training=False)
        action_log.append(action)
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            break
    stats = env.get_stats()
    stats['action_log'] = action_log
    return stats


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: RESULTS VISUALISATION
# ═══════════════════════════════════════════════════════════════════

def plot_results(rl, bh, rewards, values, losses, ticker, save_path):
    """Generate and save comprehensive results dashboard."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'AI Portfolio Trading Simulator — {ticker}',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Portfolio comparison
    ax1 = fig.add_subplot(gs[0, :2])
    n   = min(len(rl['portfolio_history']), len(bh['portfolio_history']))
    ax1.plot(rl['portfolio_history'][:n], color='#2196F3', lw=2, label='RL Agent')
    ax1.plot(bh['portfolio_history'][:n], color='#FF5722', lw=2,
             ls='--', label='Buy & Hold')
    ax1.set_title('Portfolio Value — RL Agent vs Buy & Hold', fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_xlabel('Trading Days')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # 2. Metrics bar chart
    ax2    = fig.add_subplot(gs[0, 2])
    labels = ['Total\nReturn (%)', 'Sharpe\nRatio', 'Max\nDrawdown (%)']
    rl_m   = [rl['total_return_pct'],  rl['sharpe_ratio'],  -rl['max_drawdown_pct']]
    bh_m   = [bh['total_return_pct'],  bh['sharpe_ratio'],  -bh['max_drawdown_pct']]
    x, w   = np.arange(3), 0.35
    ax2.bar(x - w/2, rl_m, w, label='RL Agent',   color='#2196F3', alpha=0.85)
    ax2.bar(x + w/2, bh_m, w, label='Buy & Hold', color='#FF5722', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('Performance Metrics', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Training reward
    ax3 = fig.add_subplot(gs[1, 0])
    sm  = pd.Series(rewards).rolling(10).mean()
    ax3.plot(rewards, alpha=0.3, color='#9C27B0', lw=0.8)
    ax3.plot(sm,      color='#9C27B0', lw=2)
    ax3.set_title('Training Reward per Episode', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.grid(True, alpha=0.3)

    # 4. Training portfolio value
    ax4 = fig.add_subplot(gs[1, 1])
    smv = pd.Series(values).rolling(10).mean()
    ax4.plot(values, alpha=0.3, color='#4CAF50', lw=0.8)
    ax4.plot(smv,    color='#4CAF50', lw=2)
    ax4.axhline(10_000, color='gray', ls='--', lw=1, label='Initial $10k')
    ax4.set_title('Episode Final Portfolio Value', fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Final Value ($)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # 5. Training loss
    ax5 = fig.add_subplot(gs[1, 2])
    if losses:
        sml = pd.Series(losses).rolling(5).mean()
        ax5.plot(losses, alpha=0.3, color='#FF9800', lw=0.8)
        ax5.plot(sml,    color='#FF9800', lw=2)
    ax5.set_title('DQN Training Loss', fontweight='bold')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Loss')
    ax5.grid(True, alpha=0.3)

    # 6. Action distribution
    ax6 = fig.add_subplot(gs[2, 0])
    counts = [rl['action_log'].count(i) for i in range(3)]
    ax6.pie(counts, labels=['Hold','Buy','Sell'],
            colors=['#607D8B','#4CAF50','#F44336'],
            autopct='%1.1f%%', startangle=90)
    ax6.set_title('Action Distribution (Backtest)', fontweight='bold')

    # 7. Drawdown
    ax7  = fig.add_subplot(gs[2, 1])
    arr  = np.array(rl['portfolio_history'])
    peak = np.maximum.accumulate(arr)
    dd   = (peak - arr) / (peak + 1e-8) * 100
    ax7.fill_between(range(len(dd)), -dd, 0, color='#F44336', alpha=0.6)
    ax7.set_title('RL Agent Drawdown Over Time', fontweight='bold')
    ax7.set_xlabel('Trading Days')
    ax7.set_ylabel('Drawdown (%)')
    ax7.grid(True, alpha=0.3)

    # 8. Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    rows = [
        ['Final Value',    f"${rl['final_value']:,.2f}",       f"${bh['final_value']:,.2f}"],
        ['Total Return',   f"{rl['total_return_pct']:+.1f}%",  f"{bh['total_return_pct']:+.1f}%"],
        ['Sharpe Ratio',   f"{rl['sharpe_ratio']:.3f}",        f"{bh['sharpe_ratio']:.3f}"],
        ['Max Drawdown',   f"{rl['max_drawdown_pct']:.1f}%",   f"{bh['max_drawdown_pct']:.1f}%"],
        ['# Trades',       str(rl['num_trades']),               'N/A'],
    ]
    tbl = ax8.table(cellText=rows, colLabels=['Metric','RL Agent','Buy & Hold'],
                    cellLoc='center', loc='center', bbox=[0,0,1,1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#ECEFF1')
    ax8.set_title('Summary Statistics', fontweight='bold', pad=10)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"📊 Chart saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  🤖 AI Portfolio Investment Simulator — Reinforcement Learning")
    print("=" * 65)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # Download data
    all_data = download_stock_data(
        CONFIG["tickers"], CONFIG["start_date"], CONFIG["end_date"])

    if not all_data:
        print("❌ No data downloaded. Check internet connection.")
        sys.exit(1)

    all_results = {}

    for ticker, df in all_data.items():
        print(f"\n{'='*65}\n  📈 Stock: {ticker}\n{'='*65}")

        train_df, _, test_df = split_data(df)

        # Environments
        train_env = PortfolioTradingEnv(
            df=train_df,
            initial_balance  = CONFIG["initial_balance"],
            window_size      = CONFIG["window_size"],
            transaction_cost = CONFIG["transaction_cost"],
        )
        test_env = PortfolioTradingEnv(
            df=test_df,
            initial_balance  = CONFIG["initial_balance"],
            window_size      = CONFIG["window_size"],
            transaction_cost = CONFIG["transaction_cost"],
        )

        state_dim = train_env.observation_space.shape[0]

        # Agent
        agent = DQNAgent(
            state_dim        = state_dim,
            action_dim       = 3,
            lr               = CONFIG["learning_rate"],
            gamma            = CONFIG["gamma"],
            epsilon          = CONFIG["epsilon"],
            epsilon_min      = CONFIG["epsilon_min"],
            epsilon_decay    = CONFIG["epsilon_decay"],
            batch_size       = CONFIG["batch_size"],
            target_update_freq = CONFIG["target_update"],
        )

        # Train
        rewards, values, losses = train_agent(
            agent, train_env, CONFIG["num_episodes"])

        # Save model
        model_path = os.path.join(CONFIG["save_dir"], f"{ticker}_model.pth")
        agent.save(model_path)

        # Backtest
        print(f"\n🔍 Backtesting on {ticker} test data...")
        agent.epsilon = 0.0
        rl_stats = run_backtest(agent, test_env)
        bh_stats = compute_bh_stats(test_df, CONFIG["initial_balance"])

        # Print comparison
        print(f"\n{'─'*52}")
        print(f"  RESULTS — {ticker}")
        print(f"  {'Metric':<20} {'RL Agent':>12} {'Buy&Hold':>12}")
        print(f"  {'─'*44}")
        for label, rk, bk in [
            ("Final Value",   f"${rl_stats['final_value']:,.2f}",      f"${bh_stats['final_value']:,.2f}"),
            ("Total Return",  f"{rl_stats['total_return_pct']:+.1f}%", f"{bh_stats['total_return_pct']:+.1f}%"),
            ("Sharpe Ratio",  f"{rl_stats['sharpe_ratio']:.3f}",       f"{bh_stats['sharpe_ratio']:.3f}"),
            ("Max Drawdown",  f"{rl_stats['max_drawdown_pct']:.1f}%",  f"{bh_stats['max_drawdown_pct']:.1f}%"),
            ("# Trades",      str(rl_stats['num_trades']),              "N/A"),
        ]:
            print(f"  {label:<20} {rk:>12} {bk:>12}")
        print(f"{'─'*52}")

        # Plot
        chart_path = os.path.join(CONFIG["save_dir"], f"{ticker}_results.png")
        plot_results(rl_stats, bh_stats, rewards, values,
                     losses, ticker, save_path=chart_path)

        all_results[ticker] = {"rl": rl_stats, "bh": bh_stats}

    # Final summary
    print(f"\n{'='*65}")
    print("  🏁 FINAL SUMMARY")
    print(f"  {'Ticker':<8} {'RL Return':>12} {'BH Return':>12} {'Sharpe':>10}")
    print(f"  {'─'*44}")
    for t, r in all_results.items():
        print(f"  {t:<8} "
              f"{r['rl']['total_return_pct']:>+10.1f}%  "
              f"{r['bh']['total_return_pct']:>+10.1f}%  "
              f"{r['rl']['sharpe_ratio']:>10.3f}")
    print(f"\n✅ Outputs saved to: {CONFIG['save_dir']}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
