"""回测引擎与绩效评估"""

import numpy as np
import pandas as pd
from . import config


def run_backtest(df: pd.DataFrame,
                 initial_capital: float = config.INITIAL_CAPITAL) -> pd.DataFrame:
    """向量化回测

    假设：每日收盘按信号调仓，仓位在下一交易日生效，
    手续费 = 成交额 * 手续费率，滑点按固定点数扣除。
    """
    out = df.dropna(subset=["position"]).reset_index(drop=True).copy()

    price = out["close"].values
    position = out["position"].values.astype(float)  # 1=多, -1=空, 0=空
    n = len(out)

    # 每手合约价值变化
    contract_value = price * config.CONTRACT_MULTIPLIER
    daily_pnl = np.zeros(n)
    costs = np.zeros(n)

    for i in range(1, n):
        # 持仓盈亏 = 上一日仓位 * 价格变动 * 乘数 * 手数
        daily_pnl[i] = position[i - 1] * (price[i] - price[i - 1]) * \
                       config.CONTRACT_MULTIPLIER * config.POSITION_SIZE

        # 调仓成本（仅在仓位变化时）
        if position[i] != position[i - 1]:
            turnover = abs(position[i] - position[i - 1]) * contract_value[i] * config.POSITION_SIZE
            costs[i] = turnover * config.COMMISSION_RATE + \
                       abs(position[i] - position[i - 1]) * config.SLIPPAGE * config.CONTRACT_MULTIPLIER

    out["daily_pnl"] = daily_pnl - costs
    out["cumulative_pnl"] = out["daily_pnl"].cumsum()
    out["equity"] = initial_capital + out["cumulative_pnl"]
    out["strategy_return"] = out["equity"].pct_change().fillna(0)
    out["benchmark_return"] = out["close"].pct_change().fillna(0)
    out["cum_strategy"] = (1 + out["strategy_return"]).cumprod()
    out["cum_benchmark"] = (1 + out["benchmark_return"]).cumprod()
    return out


def calc_metrics(df: pd.DataFrame,
                 initial_capital: float = config.INITIAL_CAPITAL) -> dict:
    """计算绩效指标"""
    rets = df["strategy_return"]
    cum = df["cum_strategy"]

    total_return = cum.iloc[-1] - 1
    n_days = len(df)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_vol = rets.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_drawdown = drawdown.min()

    # 胜率
    trades = df[df["position"] != df["position"].shift(1).fillna(0)]
    win_trades = (trades["daily_pnl"] > 0).sum()
    total_trades = len(trades)
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    # 盈亏比
    wins = trades[trades["daily_pnl"] > 0]["daily_pnl"]
    losses = trades[trades["daily_pnl"] < 0]["daily_pnl"]
    profit_loss_ratio = (wins.mean() / abs(losses.mean())) if len(losses) > 0 else np.inf

    return {
        "总收益率(%)": round(total_return * 100, 2),
        "年化收益率(%)": round(annual_return * 100, 2),
        "年化波动率(%)": round(annual_vol * 100, 2),
        "夏普比率": round(sharpe, 3),
        "最大回撤(%)": round(max_drawdown * 100, 2),
        "交易次数": total_trades,
        "胜率(%)": round(win_rate * 100, 2),
        "盈亏比": round(profit_loss_ratio, 2) if np.isfinite(profit_loss_ratio) else float("inf"),
        "期末权益": round(df["equity"].iloc[-1], 2),
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 40)
    print("回测绩效指标")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"{k:<14}: {v}")
    print("=" * 40 + "\n")
