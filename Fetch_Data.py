import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
from tqdm import tqdm

# Step 1: 获取 S&P 500 股票列表
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]
    return df['Symbol'].tolist()

# Step 2: 批量下载数据，提取指定字段
def download_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = [
            f'{ticker}_Opening price',
            f'{ticker}_Highest price',
            f'{ticker}_Lowest price',
            f'{ticker}_Closing price',
            f'{ticker}_Volume'
        ]
        
        # 用变量避免 df[column] 出现 DataFrame 问题
        closing = df[f'{ticker}_Closing price']
        volume = df[f'{ticker}_Volume']

        df[f'{ticker}_Turnover'] = closing * volume
        df[f'{ticker}_Ups and downs'] = closing.diff().fillna(0)
        df[f'{ticker}_Change'] = df[f'{ticker}_Ups and downs'] / closing.shift(1).fillna(1)
        
        return df
    
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# Step 3: 主流程
def main():
    tickers = get_sp500_tickers()
    start = datetime.today() - timedelta(days=365)
    end = datetime.today()
    combined_df = pd.DataFrame()

    for ticker in tqdm(tickers, desc="Downloading S&P 500", ncols=100):
        df = download_stock_data(ticker, start, end)
        if df is not None:
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
        time.sleep(1)


    combined_df.index.name = 'Date'
    os.makedirs('./data', exist_ok=True)
    combined_df.to_csv('./data/sp500.csv')
    print("✅ Saved to ./data/sp500.csv")

if __name__ == '__main__':
    main()
    print("📈 Data fetching completed.")