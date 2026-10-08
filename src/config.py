"""项目全局配置"""

# 数据参数
DATA_PATH = "data/soda_ash_futures.csv"
DATE_COL = "date"
PRICE_COL = "close"

# 回测参数
INITIAL_CAPITAL = 1_000_000.0       # 初始资金 100 万
COMMISSION_RATE = 0.0003            # 手续费率（双边万分之三）
SLIPPAGE = 1.0                      # 滑点（元/吨）
CONTRACT_MULTIPLIER = 20            # 纯碱期货合约乘数：20 吨/手
MARGIN_RATE = 0.12                  # 保证金比例 12%

# 模型参数
LOOKBACK = 20                       # 回看窗口
TRAIN_RATIO = 0.7                   # 训练集占比
LAG_LIST = [1, 2, 3, 5, 10]         # 滞后特征
ROLLING_WINDOWS = [5, 10, 20]       # 滚动窗口
FORECAST_HORIZON = 1                # 预测步长

# 策略参数
SIGNAL_THRESHOLD = 0.0              # 预测收益率阈值
POSITION_SIZE = 1                   # 每次交易手数
MAX_POSITION = 5                    # 最大持仓手数
STOP_LOSS_PCT = 0.02                # 止损比例 2%
TAKE_PROFIT_PCT = 0.05              # 止盈比例 5%

# 输出路径
RESULTS_DIR = "results"
