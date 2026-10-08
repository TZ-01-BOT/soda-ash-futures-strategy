"""生成 README 首图（深色量化风格净值曲线 + 关键指标）"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.features import build_features, make_target
from src.model import walk_forward_predict
from src.strategy import generate_signals, apply_risk_management
from src.backtest import run_backtest, calc_metrics

# 跑完整回测流程
df = load_data()
df = make_target(build_features(df))
df = walk_forward_predict(df)
df = apply_risk_management(generate_signals(df))
df = run_backtest(df)
m = calc_metrics(df)

# 简洁风格首图：净值曲线 + 关键指标
fig, ax = plt.subplots(figsize=(12, 5.2), dpi=130)
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

x = df["date"]
ax.plot(x, df["cum_strategy"], color="#00d98b", lw=2.2, label="Strategy")
ax.plot(x, df["cum_benchmark"], color="#58a6ff", lw=1.4, alpha=0.75, label="Buy & Hold")
ax.fill_between(x, 1, df["cum_strategy"], color="#00d98b", alpha=0.08)

ax.set_title("Time-Series-Enhanced Linear Regression Strategy\nSoda Ash Futures (SA) Daily Backtest",
             color="#e6edf3", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", frameon=False, labelcolor="#8b949e", fontsize=10)
ax.grid(color="#21262d", alpha=0.5)
ax.tick_params(colors="#8b949e", labelsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#21262d")

info = (f"Annual Return  {m['年化收益率(%)']:.2f}%     "
        f"Sharpe  {m['夏普比率']:.2f}     "
        f"Max DD  {m['最大回撤(%)']:.2f}%     "
        f"Win Rate  {m['胜率(%)']:.1f}%")
ax.text(0.012, 0.03, info, transform=ax.transAxes, color="#00d98b",
        fontsize=10.5, fontfamily="monospace")

plt.tight_layout()
plt.savefig("results/readme_banner.png", bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved results/readme_banner.png")