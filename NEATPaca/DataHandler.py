import os
import datetime as dt
import pandas as pd
import talib as ta
import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import crypto
# pip install C:\Users\jonat\OneDrive\Desktop\Data_2024\TA-lib\TA_Lib-0.4.24-cp39-cp39-win_amd64.whl

class DataHandler:
    def __init__(self, TICKER, TFRAMES, from_year=None, from_month=None, from_day=None):
        self.TICKER = TICKER
        self.TFRAMES = TFRAMES
        self.from_year = from_year
        self.from_month = from_month
        self.from_day = from_day
        self.client = CryptoHistoricalDataClient()
        self.folder_name = "./price_data"

        self.name = self.create_filename()
        
        self.df = None
        self.df_ta = None
        self.datetimes_df_ta = None
        self.last_row_df_ta = None

    def create_filename(self):
        ticker_sanitized = self.TICKER.replace("/", "-")
        tframes_joined = '-'.join(self.TFRAMES)
        self.name = f"{ticker_sanitized}_{tframes_joined}_{self.from_year}_{self.from_month}_{self.from_day}"
        return self.name
    
    def dl_price_data(self):
        # print('Downloading price data...')
        today = str(dt.datetime.today())[:-7]
        price_df = pd.DataFrame(columns=['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        dt_end = dt.datetime.strptime(today, '%Y-%m-%d %H:%M:%S')
        dt_start = dt_end + dt.timedelta(days=-7)
        end_date = self.format_date(dt_end)
        start_date = self.format_date(dt_start)

        while dt_start > dt.datetime(self.from_year, self.from_month, self.from_day):
            data = self.get_historical_data(start_date, end_date)
            dt_end = dt_end + dt.timedelta(days=-7)
            dt_start = dt_end + dt.timedelta(days=-7)
            end_date = self.format_date(dt_end)
            start_date = self.format_date(dt_start)

            price_df = pd.concat([data, price_df]).reset_index(drop=True)

        return price_df
    
    def get_historical_data(self, start_date, end_date):
        request_params = CryptoBarsRequest(
            symbol_or_symbols=self.TICKER,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date
        )

        bars = self.client.get_crypto_bars(request_params)
        dict_rows = [{k: v for k, v in dict(item).items()} for item in bars[self.TICKER]]

        hist_data = pd.DataFrame(dict_rows)
        hist_data = hist_data.drop(['trade_count', 'vwap'], axis=1)
        hist_data = hist_data.rename(columns={'timestamp': 'Date_Time',
                                            'open': 'Open',
                                            'high': 'High',
                                            'low': 'Low',
                                            'close': 'Close',
                                            'volume': 'Volume'})

        return hist_data
    
    def format_date(self, date_obj):
        return date_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

    def convert_df_by_timeframe(self, df, timeframe):
        df.Date_Time = pd.to_datetime(df.Date_Time, format='%Y-%m-%d %H:%M')
        df = df.set_index('Date_Time')
        new_df = df.groupby(pd.Grouper(freq=timeframe)).agg({'Open': 'first',
                                                            'High': 'max',
                                                            'Low': 'min',
                                                            'Close': 'last',
                                                            'Volume': 'sum'})
        new_df = new_df.reset_index()
        return new_df


    def remove_first_instance_of_consecutive_nans(self, df):
        is_nan = df.isna().all(axis=1)
        first_instance = None
        for i in range(1, len(is_nan)):
            if is_nan.iloc[i] and not is_nan.iloc[i - 1]:
                first_instance = i
                break

        if first_instance is not None:
            df = df.drop(df.index[first_instance])

        df = df.ffill()

        return df


    def calculate_ma_state(self, fast, medium, slow):
        if np.isnan(fast) or np.isnan(medium) or np.isnan(slow):
            return np.nan 
        if fast < medium < slow:
            return -0.75
        elif fast > medium > slow:
            return 0.75
        elif fast < slow < medium:
            return -0.5
        elif medium < fast < slow:
            return 0.25
        elif medium < slow < fast:
            return 0.5
        elif slow < fast < medium:
            return -0.25
        elif slow < medium < fast:
            return 1
        elif fast > slow > medium:
            return -1
        return 0


    def calculate_slope(self, series):
        return np.polyfit(range(3), series[-3:], 1)[0]

    def calculate_convergence_divergence(self, fast, medium, slow):
        current_gap_fm = abs(fast - medium)
        current_gap_fs = abs(fast - slow)
        current_gap_ms = abs(medium - slow)
        avg_gap = (current_gap_fm + current_gap_fs + current_gap_ms) / 3
        return avg_gap
    
    def add_ta_indicators(self, df_agg):
        df_timeframes_list = []

        for timeframe in self.TFRAMES:
            df_tf = self.convert_df_by_timeframe(df_agg, timeframe)
            df_tf = df_tf.dropna()

            hi = df_tf['High'].values
            lo = df_tf['Low'].values
            cl = df_tf['Close'].values
            vo = df_tf['Volume'].values

            # Create a pandas Series from 'cl' for percentage change calculation
            cl_series = pd.Series(cl, index=df_tf.index)

            # Moving Averages
            ma_fast = ta.SMA(cl, 7)
            ma_medium = ta.SMA(cl, 28)
            ma_slow = ta.SMA(cl, 99)

            # Assign MA states
            ma_states = np.vectorize(self.calculate_ma_state)(ma_fast, ma_medium, ma_slow)
            df_tf[f'MA_State_{timeframe}'] = ma_states

            # Calculate and assign slope for each MA
            df_tf[f'MA_Fast_Slope_{timeframe}'] = pd.Series(ma_fast).rolling(window=3).apply(self.calculate_slope, raw=True)
            df_tf[f'MA_Medium_Slope_{timeframe}'] = pd.Series(ma_medium).rolling(window=3).apply(self.calculate_slope, raw=True)
            df_tf[f'MA_Slow_Slope_{timeframe}'] = pd.Series(ma_slow).rolling(window=3).apply(self.calculate_slope, raw=True)

            # Calculate and assign convergence/divergence
            convergence_divergence = np.vectorize(self.calculate_convergence_divergence)(ma_fast, ma_medium, ma_slow)
            df_tf[f'MA_Conv_Div_{timeframe}'] = convergence_divergence

            # Bollinger Bands
            upperband, middleband, lowerband = ta.BBANDS(cl, 20, 2, 2)
            df_tf[f'BB_Upper_Percent_{timeframe}'] = (cl - upperband) / cl
            df_tf[f'BB_Lower_Percent_{timeframe}'] = (cl - lowerband) / cl

            # Stochastic Oscillator
            stoch_k, stoch_d = ta.STOCH(hi, lo, cl, 5, 3, 0, 3, 0)
            df_tf[f'Stoch_K_{timeframe}'] = stoch_k / 100
            df_tf[f'Stoch_D_{timeframe}'] = stoch_d / 100
            df_tf[f'Stoch_State_{timeframe}'] = (stoch_k > stoch_d).astype(int)

            # MACD
            macd, macdsignal, macdhist = ta.MACD(cl, 12, 26, 9)
            df_tf[f'MACD_State_{timeframe}'] = (macd > macdsignal).astype(int)

            # OBV
            obv = ta.OBV(cl, vo)
            df_tf['OBV'] = obv
            obv_ma = ta.SMA(obv, 20)
            df_tf[f'OBV_Trend_{timeframe}'] = (obv > obv_ma).astype(int)

            # Price % Change for the Last 24 Bars as new columns
            for i in range(1, 25):
                df_tf[f'Price_pct_change_{i}_{timeframe}'] = cl_series.pct_change(periods=i).shift(-i)

            # PSAR indicator - Binary indicator where 1 represents PSAR above close price, and 0 otherwise
            df_tf[f'PSAR_{timeframe}'] = ta.SAR(hi, lo)
            df_tf[f'PSAR_{timeframe}'] = (df_tf[f'PSAR_{timeframe}'] > cl).astype(int)

            periods = [7, 14, 28, 100]
            for period in periods:
                column_name = f'ADX_{period}_{timeframe}'
                df_tf[column_name] = ta.ADX(hi, lo, cl, period) / 100

                column_name = f'RSI_{period}_{timeframe}'
                df_tf[column_name] = ta.RSI(cl, period) / 100

                column_name = f'CCI_{period}_{timeframe}'
                df_tf[column_name] = ta.CCI(hi, lo, cl, period)
                df_tf[column_name] = (df_tf[column_name]) / 100

                column_name = f'MFI_{period}_{timeframe}'
                df_tf[column_name] = ta.MFI(hi, lo, cl, vo, period) / 100

                column_name = f'AROONOSC_{period}_{timeframe}'
                df_tf[column_name] = (ta.AROONOSC(hi, lo, period) + 100) / 200

                column_name = f'CMO_{period}_{timeframe}'
                df_tf[column_name] = (ta.CMO(cl, period) + 100) / 200

                highest_price = df_tf['High'].rolling(window=period).max()
                lowest_price = df_tf['Low'].rolling(window=period).min()
                df_tf[f'Close_to_High_Percent_{period}_{timeframe}'] = (df_tf['Close'] - highest_price) / highest_price
                df_tf[f'Close_to_Low_Percent_{period}_{timeframe}'] = (df_tf['Close'] - lowest_price) / lowest_price

            # Scaling ULTOSC values to 0-1 range, using specific time periods
            ULTOSC_periods = [(7, 14, 28), (14, 28, 56)]
            for periods in ULTOSC_periods:
                column_name = f'ULTOSC_{"_".join(map(str, periods))}_{timeframe}'
                df_tf[column_name] = ta.ULTOSC(hi, lo, cl, *periods) / 100

            # Now dropping the original price columns
            df_tf = df_tf.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'OBV'], axis=1)

            df_timeframes_list.append(df_tf)

        # Merging all dataframe timeframes
        new_df = df_timeframes_list[0]
        for idx in range(1, len(df_timeframes_list)):
            new_df = pd.merge(new_df, df_timeframes_list[idx], on='Date_Time', how='left')
            new_df.columns = new_df.columns.str.replace('_x', '')

            for col in new_df.columns:
                if '_y' in col:
                    new_df = new_df.drop([col], axis=1)

        new_df = new_df.ffill(axis=0)
        new_df = pd.merge(df_agg, new_df, on='Date_Time', how='left')
        new_df = self.remove_first_instance_of_consecutive_nans(new_df)
        new_df = new_df.dropna()

        return new_df
    
    def aggregate_data(self):
        df_agg = self.df.copy()

        # Ensure 'Date_Time' is a datetime type and set as index
        df_agg['Date_Time'] = pd.to_datetime(df_agg['Date_Time'])
        df_agg.set_index('Date_Time', inplace=True)

        # Resample and aggregate based on first timeframe
        df_agg = df_agg.resample(self.TFRAMES[0], label='right', closed='right').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index()

        return df_agg
    
    def create_folder(self):
        """Creates a folder if it does not exist."""
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

    def get_data(self):
        self.create_folder()
        self.df = self.dl_price_data()
        df_agg = self.aggregate_data()
        self.df_ta = self.add_ta_indicators(df_agg)
        self.df_ta.to_csv(self.folder_name + '/' + self.name + '.csv', index=False)

    def update_data(self):
        self.datetimes_df_ta = self.df_ta['Date_Time'].values
        request_params = crypto.CryptoLatestBarRequest(symbol_or_symbols=self.TICKER)
        latest_bar = self.client.get_crypto_latest_bar(request_params)
        bar = latest_bar[self.TICKER]

        new_row = {
            'symbol': bar.symbol,
            'Date_Time': bar.timestamp,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        }

        new_df = pd.DataFrame([new_row])

        if not self.df.empty and 'Date_Time' in self.df.columns and self.df['Date_Time'].iloc[-1] != new_df['Date_Time'].iloc[0]:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df = self.df.drop(self.df.index[0])

            if 'Date_Time' in self.df.columns:
                df_agg = self.aggregate_data()
                self.df_ta = self.add_ta_indicators(df_agg)
                current_last_df_ta_row = self.df_ta.tail(1)['Date_Time'].values[0]
                if self.last_row_df_ta is None or self.last_row_df_ta['Date_Time'].values[0] != current_last_df_ta_row:

                    self.last_row_df_ta = self.df_ta.tail(1)
                    self.datetimes_df_ta = self.df_ta['Date_Time'].values


if __name__ == "__main__":
    handler = DataHandler(TICKER="ETH/USD", 
                          TFRAMES=['30Min', '2H'], 
                          from_year=2023, 
                          from_month=10, 
                          from_day=1)
    
    handler.get_data()
    print(f"{handler.name}\tMinute Dataframe")
    print(handler.df)
    print(f"\n\n{handler.name}\tTA Dataframe")
    print(handler.df_ta)