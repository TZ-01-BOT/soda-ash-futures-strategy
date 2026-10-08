"""数据加载与预处理模块"""

import os
import numpy as np
import pandas as pd
from . import config


def load_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    """加载期货日线数据，要求包含 date, open, high, low, close, volume 列"""
    if not os.path.exists(path):
        print(f"[提示] 未找到数据文件 {path}，使用合成样本数据进行演示。")
        return generate_sample_data()

    df = pd.read_csv(path, parse_dates=[config.DATE_COL])
    df = df.sort_values(config.DATE_COL).reset_index(drop=True)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")

    # 去除异常值与停牌日
    df = df[(df["close"] > 0) & (df["volume"] > 0)].reset_index(drop=True)
    return df


def generate_sample_data(periods: int = 1500, seed: int = 42) -> pd.DataFrame:
    """生成合成的纯碱期货日线数据，用于演示与测试"""
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=periods)
    n = len(dates)

    # 用带均值回归的几何布朗运动模拟价格
    base_price = 2200.0
    kappa = 0.02          # 均值回归速度
    theta = np.log(base_price)
    sigma = 0.015         # 日波动率

    log_price = np.zeros(n)
    log_price[0] = np.log(base_price)
    for i in range(1, n):
        shock = rng.normal(0, sigma)
        log_price[i] = log_price[i - 1] + kappa * (theta - log_price[i - 1]) + shock

    close = np.exp(log_price)
    # 生成 OHLC
    intraday_vol = sigma * close
    high = close + np.abs(rng.normal(0, intraday_vol))
    low = close - np.abs(rng.normal(0, intraday_vol))
    open_ = close * (1 + rng.normal(0, sigma * 0.3, size=n))
    volume = rng.integers(50_000, 500_000, size=n)

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(open_, 1),
        "high": np.round(high, 1),
        "low": np.round(low, 1),
        "close": np.round(close, 1),
        "volume": volume,
    })
    return df
