"""项目入口：数据加载 → 特征工程 → 模型预测 → 策略生成 → 回测 → 可视化"""

import argparse
from src.data_loader import load_data
from src.features import build_features, make_target
from src.model import walk_forward_predict
from src.strategy import generate_signals, apply_risk_management
from src.backtest import run_backtest, calc_metrics, print_metrics
from src.visualization import plot_results


def main():
    parser = argparse.ArgumentParser(description="纯碱期货时间序列增强线性回归策略回测")
    parser.add_argument("--data", type=str, default=None, help="CSV 数据文件路径")
    args = parser.parse_args()

    # 1. 加载数据
    df = load_data(args.data) if args.data else load_data()
    print(f"[数据] 加载完成，共 {len(df)} 条记录，"
          f"区间 {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    # 2. 特征工程 + 目标变量
    df = build_features(df)
    df = make_target(df)
    print(f"[特征] 特征工程完成，特征数: {len(df.columns) - 7}")

    # 3. 滚动窗口预测
    df = walk_forward_predict(df)
    valid = df.dropna(subset=["prediction"])
    print(f"[模型] 滚动预测完成，有效预测样本: {len(valid)}")

    # 4. 生成交易信号 + 风控
    df = generate_signals(df)
    df = apply_risk_management(df)

    # 5. 回测
    df = run_backtest(df)
    metrics = calc_metrics(df)
    print_metrics(metrics)

    # 6. 可视化
    plot_results(df, metrics)
    print("[完成] 回测流程结束。")


if __name__ == "__main__":
    main()
