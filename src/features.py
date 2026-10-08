"""时间序列特征工程模块

核心思想：用时间序列特征（滞后项、滚动统计量、技术指标）增强线性回归模型，
使原本只能捕捉线性关系的 OLS 能够利用价格序列中的时序依赖信息。
"""

import numpy as np
import pandas as pd
from . import config


def add_lag_features(df: pd.DataFrame, col: str = config.PRICE_COL,
                     lags: list = None) -> pd.DataFrame:
    """添加滞后项特征"""
    lags = lags or config.LAG_LIST
    out = df.copy()
    for lag in lags:
        out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, col: str = config.PRICE_COL,
                         windows: list = None) -> pd.DataFrame:
    """添加滚动统计特征：均值、标准差、最大值、最小值、收益率"""
    windows = windows or config.ROLLING_WINDOWS
    out = df.copy()
    for w in windows:
        out[f"{col}_ma{w}"] = out[col].rolling(w).mean()
        out[f"{col}_std{w}"] = out[col].rolling(w).std()
        out[f"{col}_max{w}"] = out[col].rolling(w).max()
        out[f"{col}_min{w}"] = out[col].rolling(w).min()
        out[f"{col}_ret{w}"] = out[col].pct_change(w)
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加常见技术指标：RSI、MACD、布林带、ATR、量价"""
    out = df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # 布林带
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["boll_upper"] = ma20 + 2 * std20
    out["boll_lower"] = ma20 - 2 * std20
    out["boll_pct"] = (close - out["boll_lower"]) / (out["boll_upper"] - out["boll_lower"])

    # ATR (平均真实波幅)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()

    # 量价特征
    out["vol_ratio"] = volume / volume.rolling(20).mean()
    out["log_return"] = np.log(close / close.shift())

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建完整特征集"""
    out = df.copy()
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_technical_indicators(out)
    return out


def make_target(df: pd.DataFrame, horizon: int = config.FORECAST_HORIZON) -> pd.DataFrame:
    """生成预测目标：未来 horizon 日的对数收益率"""
    out = df.copy()
    out["target"] = np.log(out[config.PRICE_COL].shift(-horizon) / out[config.PRICE_COL])
    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取用于训练的特征列名（排除原始行情列与目标列）"""
    exclude = {"date", "open", "high", "low", "close", "volume", "target"}
    return [c for c in df.columns if c not in exclude]
