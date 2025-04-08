import pandas as pd
import numpy as np

class StockDataPreprocessor:
    def __init__(self, csv_path, time_steps=5):
        """
        初始化
        :param csv_path: CSV文件路径
        :param time_steps: 用于滑动窗口的时间步数
        """
        self.csv_path = csv_path
        self.time_steps = time_steps
        self.df = pd.read_csv(csv_path)
        self.feature_df = self.df.iloc[:, 1:]  # 排除时间列
        self.stock_names = self._extract_stock_names()

    def _extract_stock_names(self):
        """提取所有股票的名称（从列名前缀）"""
        column_names = self.df.columns[1:]  # 跳过时间列
        return sorted(set(col.split('_')[0] for col in column_names))

    def _standardize(self, data):
        """对每支股票数据做 z-score 标准化"""
        return (data - data.mean()) / data.std()

    def _extract_samples_for_stock(self, stock):
        """对某支股票提取滑动窗口样本"""
        stock_cols = [col for col in self.feature_df.columns if col.startswith(f"{stock}_")] # get the real column names(pair with slash)
        stock_data = self.feature_df[stock_cols].copy()
        # stock_data = self._standardize(stock_data).dropna()

        X_stock, y_stock = [], []
        for i in range(len(stock_data) - self.time_steps):
            window = stock_data.iloc[i:i + self.time_steps].values
            target = stock_data.iloc[i + self.time_steps][f"{stock}_Closing price"]
            X_stock.append(window)
            y_stock.append(target)
        # print(f"{stock}: {window.shape}")

        return X_stock, y_stock

    def preprocess(self):
        """整体预处理，返回所有股票合并后的 X, y"""
        X_all, y_all = [], []
        for stock in self.stock_names:
            X_stock, y_stock = self._extract_samples_for_stock(stock)
            X_all.extend(X_stock)
            y_all.extend(y_stock)
        return np.array(X_all), np.array(y_all)
    
    def plot_stock_candlestick(self, stock_name):
        """
        绘制指定股票的蜡烛图（开盘价、收盘价、最高价、最低价）
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        # 提取时间和该股票的价格数据
        time = pd.to_datetime(self.df.iloc[:, 0])  # 时间列
        columns = [f"{stock_name}_Opening price", f"{stock_name}_Highest price",
                f"{stock_name}_Lowest price", f"{stock_name}_Closing price"]

        if not all(col in self.df.columns for col in columns):
            print(f"股票 {stock_name} 的价格列缺失，跳过")
            return

        data = self.df[columns].copy()
        data["Date"] = time
        data = data.dropna()

        # 开始绘制K线图
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, row in data.iterrows():
            color = 'red' if row[f"{stock_name}_Closing price"] >= row[f"{stock_name}_Opening price"] else 'green'
            # 主体（矩形框）
            rect = Rectangle(
                (row["Date"], min(row[f"{stock_name}_Opening price"], row[f"{stock_name}_Closing price"])),
                width=pd.Timedelta(days=0.8),
                height=abs(row[f"{stock_name}_Opening price"] - row[f"{stock_name}_Closing price"]),
                color=color
            )
            ax.add_patch(rect)
            # 上影线 / 下影线
            ax.plot([row["Date"], row["Date"]],
                    [row[f"{stock_name}_Lowest price"], row[f"{stock_name}_Highest price"]],
                    color=color)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_title(f"{stock_name} K line Chart")
        ax.set_ylabel("price")
        ax.set_xlabel("Date")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 示例使用
    csv_path = ".\\data\\sp500.csv"  # 替换为你的 CSV 文件路径
    preprocessor = StockDataPreprocessor(csv_path, time_steps=5)
    X, y = preprocessor.preprocess()
    preprocessor.plot_stock_candlestick("AAPL")  # 绘制苹果公司的K线图
    print(f"X shape: {X.shape}, y shape: {y.shape}") # X[0:N1] 为第1只股票的样本，X[N1:N2] 为第2只股票的样本，以此类推
    