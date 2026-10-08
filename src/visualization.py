"""可视化模块：净值曲线、回撤、信号图"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from . import config


def _has_cjk_font() -> bool:
    """检测系统是否有可用的中文字体"""
    cjk_keywords = ["SimHei", "Heiti", "WenQuanYi", "Noto Sans CJK", "PingFang", "Microsoft YaHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    return any(any(k in name for k in cjk_keywords) for name in available)


def plot_results(df: pd.DataFrame, metrics: dict, save_dir: str = config.RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)

    # 根据系统字体自动选择中英文标签
    use_cjk = _has_cjk_font()
    if use_cjk:
        plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        l_strat, l_bh = "策略净值", "买入持有"
        t1 = "策略净值 vs 买入持有"
        t2 = f"策略回撤 (最大回撤 {metrics['最大回撤(%)']}%)"
        l_long, l_short = "做多", "做空"
        t3 = "纯碱期货价格与持仓信号"
    else:
        l_strat, l_bh = "Strategy", "Buy & Hold"
        t1 = "Equity Curve: Strategy vs Buy & Hold"
        t2 = f"Drawdown (Max DD {metrics['最大回撤(%)']}%)"
        l_long, l_short = "Long", "Short"
        t3 = "Soda Ash Futures Price & Position Signals"

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. 净值曲线
    axes[0].plot(df["date"], df["cum_strategy"], label=l_strat, color="#e74c3c")
    axes[0].plot(df["date"], df["cum_benchmark"], label=l_bh, color="#3498db", alpha=0.6)
    axes[0].set_title(t1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. 回撤
    cum = df["cum_strategy"]
    drawdown = (cum - cum.cummax()) / cum.cummax()
    axes[1].fill_between(df["date"], drawdown, 0, color="#e74c3c", alpha=0.4)
    axes[1].set_title(t2)
    axes[1].grid(alpha=0.3)

    # 3. 价格与交易信号
    axes[2].plot(df["date"], df["close"], color="black", linewidth=0.8)
    long_mask = df["position"] > 0
    short_mask = df["position"] < 0
    axes[2].scatter(df.loc[long_mask, "date"], df.loc[long_mask, "close"],
                    c="red", s=8, label=l_long)
    axes[2].scatter(df.loc[short_mask, "date"], df.loc[short_mask, "close"],
                    c="green", s=8, label=l_short)
    axes[2].set_title(t3)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "backtest_result.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[可视化] 结果已保存至 {path}")
    return path
