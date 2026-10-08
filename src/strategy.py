"""交易策略模块

基于模型对未来收益率的预测生成交易信号：
- 预测收益率 > 阈值 → 做多
- 预测收益率 < -阈值 → 做空
- 否则 → 平仓
结合止损、止盈与仓位管理。
"""

import numpy as np
import pandas as pd
from . import config


def generate_signals(df: pd.DataFrame, threshold: float = config.SIGNAL_THRESHOLD) -> pd.DataFrame:
    """根据预测收益率生成交易信号

    信号取值: 1=做多, -1=做空, 0=空仓
    """
    out = df.copy()
    out["signal"] = 0
    out.loc[out["prediction"] > threshold, "signal"] = 1
    out.loc[out["prediction"] < -threshold, "signal"] = -1
    out["signal"] = out["signal"].fillna(0).astype(int)
    return out


def apply_risk_management(df: pd.DataFrame) -> pd.DataFrame:
    """应用止损止盈：在持仓期间，若价格触及止损/止盈线则平仓"""
    out = df.copy()
    out["position"] = out["signal"].shift(1).fillna(0)  # 次日开盘执行

    entry_price = None
    entry_position = 0

    for i in range(1, len(out)):
        pos = out.loc[i, "position"]
        price = out.loc[i, "close"]

        if entry_position != 0 and pos != entry_position:
            # 平仓
            entry_price = None
            entry_position = 0

        if pos != 0 and entry_price is None:
            entry_price = price
            entry_position = pos
            continue

        if entry_position != 0 and entry_price is not None:
            ret = (price - entry_price) / entry_price * entry_position
            if ret <= -config.STOP_LOSS_PCT or ret >= config.TAKE_PROFIT_PCT:
                out.loc[i, "position"] = 0
                out.loc[i, "signal"] = 0
                entry_price = None
                entry_position = 0

    return out
