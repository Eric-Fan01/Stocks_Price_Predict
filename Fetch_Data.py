import os
import time
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import pickle
import re


class SP500Downloader:
    def __init__(self, save_path='./data/sp500.csv', days=365, sleep_time=1):
        self.save_path = save_path
        self.days = days
        self.sleep_time = sleep_time
        self.tickers = self.get_sp500_tickers()
        self.start = datetime.today() - timedelta(days=self.days)
        self.end = datetime.today()
        self.combined_df = pd.DataFrame()

    def get_sp500_tickers(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        df = table[0]
        return df['Symbol'].tolist()

    def download_stock_data(self, ticker):
        try:
            df = yf.download(ticker, start=self.start, end=self.end, progress=False)
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

            closing = df[f'{ticker}_Closing price']
            volume = df[f'{ticker}_Volume']

            df[f'{ticker}_Turnover'] = closing * volume
            df[f'{ticker}_Ups and downs'] = closing.diff().fillna(0)
            df[f'{ticker}_Change'] = df[f'{ticker}_Ups and downs'] / closing.shift(1).fillna(1)

            return df

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    def run(self):
        for ticker in tqdm(self.tickers, desc="Downloading S&P 500", ncols=100):
            df = self.download_stock_data(ticker)
            if df is not None:
                if self.combined_df.empty:
                    self.combined_df = df
                else:
                    self.combined_df = self.combined_df.join(df, how='outer')
            time.sleep(self.sleep_time)

        self.combined_df.index.name = 'Date'
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.combined_df.to_csv(self.save_path)
        print(f"✅ Saved to {self.save_path}")



class SP500Formatter:
    def __init__(self, csv_path, output_path='./data/sp500.p'):
        self.csv_path = csv_path
        self.output_path = output_path
        self.df_wide = None
        self.df_long = None

    def load_csv(self):
        print(f"📥 Loading CSV from {self.csv_path}")
        self.df_wide = pd.read_csv(self.csv_path, parse_dates=['Date'])
        self.df_wide.set_index('Date', inplace=True)

    def convert_to_long_format(self):
        print("🔄 Converting to long format...")
        melted = self.df_wide.stack().reset_index()
        melted.columns = ['Date', 'Ticker_Field', 'Value']

        # extract Ticker and Field from Ticker_Field
        melted['Ticker'] = melted['Ticker_Field'].apply(lambda x: re.match(r'^([A-Z.]+)', x).group(1))
        melted['Field'] = melted['Ticker_Field'].apply(lambda x: x.split('_', 1)[1])

        # 透视为长格式; pivot to long format
        self.df_long = melted.pivot_table(index=['Date', 'Ticker'], columns='Field', values='Value').reset_index()

        # ✅ 按 Ticker 再按 Date 排序，便于后续训练输入切片; rank by Ticker first and then Date, for easier slicing
        self.df_long = self.df_long.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    def save_as_pickle(self):
        print(f"💾 Saving to pickle: {self.output_path}")
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.df_long, f)
        print("✅ Saved sorted long-format DataFrame!")

    def run(self):
        self.load_csv()
        self.convert_to_long_format()
        self.save_as_pickle()


class SingleStockFormatterForSP500:
    def __init__(self, csv_path, ticker='AAPL', output_path='./data/aapl.p'):
        self.csv_path = csv_path
        self.ticker = ticker
        self.output_path = output_path
        self.norm_path = output_path.replace('.p', '_norm.pkl')
        self.df = None
        self.df_long = None
        self.norm_params = {}

    def load_and_clean_csv(self):
        print(f"📥 Loading raw AAPL CSV from {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        df.columns = [col.strip() for col in df.columns]

        df = df.rename(columns={
            'Date': 'Date',
            'Close/Last': 'Closing price',
            'Open': 'Opening price',
            'High': 'Highest price',
            'Low': 'Lowest price',
            'Volume': 'Volume'
        })

        # 去掉$符号 → 转 float
        for col in ['Closing price', 'Opening price', 'Highest price', 'Lowest price']:
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)

        # 补充字段
        df['Turnover'] = df['Closing price'] * df['Volume']
        df['Ups and downs'] = df['Closing price'].diff().fillna(0)
        df['Change'] = df['Ups and downs'] / df['Closing price'].shift(1).fillna(1)

        df['Ticker'] = self.ticker
        df['Date'] = pd.to_datetime(df['Date'])

        self.df = df

    def standardize_columns(self):
        print("⚙️ Standardizing numerical columns...")
        # 选取数值列（排除日期和 Ticker）
        cols_to_standardize = ['Opening price', 'Highest price', 'Lowest price',
                               'Closing price', 'Volume', 'Turnover', 'Ups and downs', 'Change']

        for col in cols_to_standardize:
            mean = self.df[col].mean()
            std = self.df[col].std()
            self.df[col] = (self.df[col] - mean) / std

            # 保存均值和标准差
            self.norm_params[col] = {'mean': mean, 'std': std}

    def convert_to_long_format(self):
        print("🔄 Converting AAPL to SP500 long format...")
        cols = ['Date', 'Ticker', 'Opening price', 'Highest price', 'Lowest price',
                'Closing price', 'Volume', 'Turnover', 'Ups and downs', 'Change']
        self.df_long = self.df[cols].copy()
        self.df_long = self.df_long.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    def save_as_pickle(self):
        print(f"💾 Saving long-format to {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.df_long, f)
        print("✅ Saved .p!")

        print(f"💾 Saving normalization params to {self.norm_path}")
        with open(self.norm_path, 'wb') as f:
            pickle.dump(self.norm_params, f)
        print("✅ Saved normalization parameters!")

    def run(self):
        self.load_and_clean_csv()
        self.standardize_columns()
        self.convert_to_long_format()
        self.save_as_pickle()


if __name__ == '__main__':

    # # fetch S&P 500 data
    # downloader = SP500Downloader(save_path='/data/sp600.csv')
    # downloader.run()
    # print("📈 Data fetching completed.")


    # # format S&P 500 data
    # formatter = SP500Formatter(csv_path='./data/sp500.csv', output_path='./data/sp500.p')
    # formatter.run()
    # print("📊 Data formatting completed.")

    # with open('./data/sp500.p', 'rb') as f:
    #     df = pickle.load(f)
    # for ticker, df_group in df.groupby('Ticker'):
    #     print(df_group.head(3))

    # apl = SingleStockFormatterForSP500(csv_path='./data/AAPL.csv', ticker='AAPL', output_path='./data/aapl.p')
    # apl.run()   

    # with open('./data/aapl.p', 'rb') as f:
    #     df = pickle.load(f)
    # print(df.head(3))
    with open('./data/aapl_norm.pkl', 'rb') as f:
        norm_params = pickle.load(f)
    print(norm_params)