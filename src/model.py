"""时间序列增强线性回归模型

"增强"体现在两点：
1. 特征层面：引入大量时间序列特征（滞后、滚动统计、技术指标），见 features.py
2. 建模层面：采用滚动窗口（Walk-Forward）训练，避免未来函数；
   并对残差做一阶自回归（AR(1)）修正，捕捉线性模型未能解释的时序依赖。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from . import config
from .features import get_feature_columns


class TSLinearRegression:
    """时间序列增强线性回归

    Parameters
    ----------
    alpha : float
        Ridge 回归正则化系数，0 表示普通 OLS。
    fit_residual_ar : bool
        是否对残差做 AR(1) 修正。
    """

    def __init__(self, alpha: float = 1.0, fit_residual_ar: bool = True):
        self.alpha = alpha
        self.fit_residual_ar = fit_residual_ar
        self.model = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.ar_coef = 0.0
        self.intercept = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        if self.fit_residual_ar:
            pred = self.model.predict(X_scaled)
            resid = y.values - pred
            # AR(1) 修正：resid_t = phi * resid_{t-1} + eps
            resid_lag = resid[:-1]
            resid_cur = resid[1:]
            if len(resid_lag) > 1 and np.var(resid_lag) > 0:
                self.ar_coef = np.cov(resid_lag, resid_cur)[0, 1] / np.var(resid_lag)
                self.intercept = np.mean(resid_cur) - self.ar_coef * np.mean(resid_lag)
        return self

    def predict(self, X: pd.DataFrame, last_resid: float = 0.0) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_cols])
        pred = self.model.predict(X_scaled)
        if self.fit_residual_ar:
            pred = pred + self.intercept + self.ar_coef * last_resid
        return pred

    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_, index=self.feature_cols).sort_values(
            key=abs, ascending=False
        )


def walk_forward_predict(df: pd.DataFrame, train_ratio: float = config.TRAIN_RATIO,
                         retrain_every: int = 60) -> pd.DataFrame:
    """滚动窗口（Walk-Forward）预测，避免未来函数

    Parameters
    ----------
    df : 已包含特征与 target 的 DataFrame
    train_ratio : 初始训练集占比
    retrain_every : 每隔多少个样本重新训练一次模型

    Returns
    -------
    带 prediction 列的 DataFrame
    """
    df = df.dropna().reset_index(drop=True)
    feature_cols = get_feature_columns(df)
    n = len(df)
    train_size = int(n * train_ratio)

    predictions = np.full(n, np.nan)
    model = TSLinearRegression()

    # 初始训练
    train_X = df.loc[:train_size - 1, feature_cols]
    train_y = df.loc[:train_size - 1, "target"]
    model.fit(train_X, train_y)

    # 获取初始残差用于 AR 修正
    train_pred = model.predict(train_X)
    last_resid = (train_y.values - train_pred)[-1]

    for i in range(train_size, n):
        X_i = df.loc[[i], feature_cols]
        predictions[i] = model.predict(X_i, last_resid=last_resid)[0]

        # 用真实值更新残差
        last_resid = df.loc[i, "target"] - predictions[i]

        # 定期重训练
        if (i - train_size) % retrain_every == 0 and i > train_size:
            model.fit(df.loc[:i, feature_cols], df.loc[:i, "target"])
            train_pred = model.predict(df.loc[:i, feature_cols])
            last_resid = (df.loc[:i, "target"].values - train_pred)[-1]

    out = df.copy()
    out["prediction"] = predictions
    return out
