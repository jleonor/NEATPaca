from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import ClosePositionRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.historical import crypto
from obj.NEATTradingAgent import TradingAgent
import datetime as dt
import pandas as pd
import numpy as np
import talib as ta
import pickle
import time
from DataHandler import *
from Logger import *
import shutup; shutup.please()

class NEATPacaTrader():
    def __init__(self, TICKERS, AGENT_GENERATIONS, TFRAMES, API_KEY, API_SECRET):
        
        self.logger = Logger(folder_path="trade_logs", 
                        file_name="trade_log",
                        level="DEBUG")

        self.logger.log('Initializing DM variables', level="DEBUG")

        self.log_messages = []
        self.decisions = []
        self.TICKERS = TICKERS
        
        self.AGENT_GENERATIONS = AGENT_GENERATIONS
        
        self.data_handlers = {}
        self.agents = {}
        self.last_rows = {}
        self.previous_datetimes = {}
        self.eval_rows = {}
        
        # Initialize the client for crypto historical data
        self.TFRAMES = TFRAMES

        # Set your API key and secret
        self.API_KEY = API_KEY
        self.API_SECRET = API_SECRET

        self.logger.log('Connecting to Alpaca', level="DEBUG")

        # Initialize the trading client
        self.trading_client = TradingClient(self.API_KEY, self.API_SECRET, paper=True)
        self.latest_client = crypto.CryptoHistoricalDataClient()
        self.data_client = CryptoHistoricalDataClient(self.API_KEY, self.API_SECRET)

        self.max_timeframe_minutes = self.parse_timeframes()
        self.required_days = self.calculate_required_days()

        self.from_year, self.from_month, self.from_day = self.adjust_start_date()

        if not os.path.exists("trading_checkpoints"):
            os.makedirs("trading_checkpoints")

        for ticker in self.TICKERS:
            self.logger.log(f'Loading NEAT agent for {ticker}')
            checkpoint_path = f"./trading_checkpoints/{ticker.replace('/USD', '')}_checkpoint.pkl"
            if os.path.exists(checkpoint_path):
                agent = self.load_checkpoint(checkpoint_path)
                self.show_agent_state(agent)
            else:
                agent = self.load_winner_agent(f"./models/{ticker.replace('/USD', '')}_winner_genome_gen_{self.AGENT_GENERATIONS[ticker]}.pkl")
                self.init_agent(agent)
                self.show_agent_state(agent)

            self.logger.log(f"Downloading {ticker} data from: {self.from_year}-{self.from_month:02d}-{self.from_day:02d} until now ({self.required_days} days of data for a {self.max_timeframe_minutes}min timeframe)")
            handler = DataHandler(TICKER=ticker, 
                        TFRAMES=self.TFRAMES, 
                        from_year=self.from_year, 
                        from_month=self.from_month, 
                        from_day=self.from_day)
            handler.get_data()
            
            self.data_handlers[ticker] = handler
            self.agents[ticker] = agent

            self.last_rows[ticker] = None
            self.previous_datetimes[ticker] = None
            self.eval_rows[ticker] = None

        self.logger.log("Initialisation complete", level="DEBUG")
            

    def calculate_required_days(self, bars=300):
        minutes_needed = self.max_timeframe_minutes * bars
        days_needed = minutes_needed / (24 * 60)
        return days_needed
    
    def parse_timeframes(self):
        timeframe_minutes = []
        for tf in self.TFRAMES:
            number, unit = int(''.join(filter(str.isdigit, tf))), ''.join(filter(str.isalpha, tf))
            if unit.startswith('Min'):
                timeframe_minutes.append(number)
            elif unit.startswith('H'):
                timeframe_minutes.append(number * 60)
        return max(timeframe_minutes)

    def buy(self, currency):
        try:
            # Get account information to find out portfolio value and cash
            account = self.trading_client.get_account()
            used_amount = float(account.portfolio_value) * 0.80
            
            # Get all positions and calculate positions_sum
            positions = self.trading_client.get_all_positions()
            positions_sum = sum(float(pos.market_value) for pos in positions if pos.symbol.replace("/", "") in self.TICKERS)
            
            # Calculate available_cash
            available_cash = used_amount - positions_sum
            
            # Determine the number of open positions for the traded currencies
            open_positions_count = sum(1 for pos in positions if pos.symbol.replace("/", "") in self.TICKERS)
            
            # Calculate the cash allocation per currency
            cash_for_currency = available_cash / (len(self.TICKERS) - open_positions_count)

            # Get the current price of the currency
            bars = self.data_client.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=currency,
                    timeframe=TimeFrame.Minute
                )
            )
            current_price = bars[currency][0].close  # Current price of ETH

            # Calculate the amount of currency to buy
            amount_to_buy = cash_for_currency / current_price

            # Place a market order to buy as much currency as possible
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=currency,
                    qty=amount_to_buy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC  # Good 'Til Cancelled
                )
            )

            return f"Buy order success: {round(amount_to_buy, 5)} {currency} | Value: ${cash_for_currency} | CurrentPrice: ${current_price} | Order ID: {order.id}"

        except Exception as e:
            return f"Failed to buy {currency} --- {e}"


    def load_checkpoint(self, filename):
        with open(filename, 'rb') as f:
            checkpoint_agent = pickle.load(f)
        return checkpoint_agent

    def save_checkpoint(self, agent, ticker):
        with open(f"./trading_checkpoints/{ticker.replace('/USD', '')}_checkpoint.pkl", 'wb') as f:
            pickle.dump(agent, f)
    
    def sell(self, currency):
        # Attempt to get the open position for currency
        try:
            position = self.trading_client.get_open_position(currency.replace("/", ""))
            currency_quantity = float(position.qty)  # Quantity of currency you own

            if currency_quantity > 0:
                # Fetch the current price of currency
                bars = self.data_client.get_crypto_bars(
                    CryptoBarsRequest(
                        symbol_or_symbols=currency,
                        timeframe=TimeFrame.Minute
                    )
                )

                current_price = bars[currency][0].close

                # Place a market order to sell currency
                order = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=currency,
                        qty=currency_quantity,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC  # Good 'Til Cancelled
                    )
                )
                return f"Sell order success: {currency_quantity} {currency} | Value: ${position.market_value} | CurrentPrice: ${current_price} | Order ID: {order.id}"

        except Exception as e:
            return f"Failed to sell {currency} --- {e}"

    def add_to_log(self, message):
        print(f"{(dt.datetime.now() - dt.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    def load_winner_agent(self, filename):
        with open(filename, 'rb') as f:
            winner_agent = pickle.load(f)
        return winner_agent

    def make_decision_on_single_row(agent, row_ta, current_price):
        decision = agent.make_decision(row_ta.to_numpy(), current_price)
        return decision

    def show_agent_state(self, agent):
        print("\n*** AGENT STATUS ***")
        print(f"Starting money: {agent.starting_money}")
        print(f"Current money: {agent.money}")
        print(f"Transaction fee percentage: {agent.transaction_fee_percent}")
        print(f"Position open: {agent.position_open}")
        print(f"Entry price: {agent.entry_price}")
        print(f"Trade log: {agent.trade_log}")
        print(f"Daily balances: {agent.daily_balances}")
        print(f"Wins: {agent.wins}")
        print(f"Losses: {agent.losses}")
        print(f"Equity peak: {agent.equity_peak}")
        print(f"Max drawdown: {agent.max_drawdown}")
        print(f"Portfolio: {agent.portfolio}")
        print(f"Last action period: {agent.last_action_period}")
        print()

    def init_agent(self, agent):
        agent.money = agent.starting_money
        agent.position_open = False
        agent.entry_price = 0
        agent.trade_log = []
        agent.daily_balances = [agent.starting_money]
        agent.wins = 0
        agent.losses = 0
        agent.equity_peak = agent.starting_money
        agent.max_drawdown = 0
        agent.portfolio = agent.starting_money
        agent.last_action_period = 0

    def make_decision_on_single_row(self, agent, row_ta, current_price):
        decision = agent.make_decision(row_ta.to_numpy(), current_price)
        return decision

    def adjust_start_date(self):
        today = dt.datetime.today()
        start_date = today - dt.timedelta(days=int(self.required_days) + 1)

        return start_date.year, start_date.month, start_date.day

    def trade(self):
        try:
            self.logger.log("Beginning trading session")
            while True:
                for ticker in self.TICKERS:

                    self.data_handlers[ticker].update_data()

                    current_datetime = dt.datetime.now()
                    current_datetime = current_datetime - dt.timedelta(hours=2)
                    current_datetime = current_datetime.replace(second=0, microsecond=0)
                    current_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

                    dt_df_ta = [pd.to_datetime(date).to_pydatetime() for date in self.data_handlers[ticker].datetimes_df_ta]
                    dt_df_ta = [date.replace(second=0, microsecond=0) for date in dt_df_ta]
                    dt_df_ta = [date.strftime('%Y-%m-%d %H:%M:%S') for date in dt_df_ta]
                    
                    if self.previous_datetimes[ticker] is None or (current_datetime in dt_df_ta and current_datetime != self.previous_datetimes[ticker]):
                        if self.previous_datetimes[ticker] is None:
                            self.previous_datetimes[ticker] = current_datetime
                            continue
                        else:
                            eval_row = self.data_handlers[ticker].df_ta.loc[self.data_handlers[ticker].df_ta['Date_Time'] == current_datetime]

                        self.previous_datetimes[ticker] = current_datetime

                        current_price = eval_row['Close'].iloc[0]
                        date_time = eval_row['Date_Time'].iloc[0]

                        eval_row = eval_row.drop(['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
                        decision = self.make_decision_on_single_row(self.agents[ticker], eval_row, current_price)

                        decision_map = {0: "BUY", 1: "NONE", 2: "SELL"}
                        self.logger.log(f"Current price for {ticker} is {current_price} | Decision was {decision_map[decision]}", level="DEBUG")
                        if decision == 0 and not self.agents[ticker].position_open:
                            buy_status = self.buy(ticker)

                            if "success" in buy_status:
                                self.logger.log(f"Order status: {buy_status}")
                                self.agents[ticker].execute_trade(decision, current_price, date_time)
                                self.agents[ticker].daily_balances.append(self.agents[ticker].portfolio)
                            else:
                                self.logger.log(f"Order status: {buy_status}", level="ERROR")

                        elif decision == 2 and self.agents[ticker].position_open:
                            sell_status = self.sell(ticker)

                            if "success" in sell_status:
                                self.logger.log(f"Order status: {sell_status}")
                                self.agents[ticker].execute_trade(decision, current_price, date_time)
                                self.agents[ticker].daily_balances.append(self.agents[ticker].portfolio)
                            else:
                                self.logger.log(f"Order status: {sell_status}", level="ERROR")
                        else:
                            self.logger.log(f"Open position for {ticker}: {self.agents[ticker].position_open}", level="DEBUG")
                        
                        self.save_checkpoint(self.agents[ticker], ticker)

                time.sleep(20)
        
        except Exception as e:
            # Save a checkpoint in case of an error
            for ticker in self.TICKERS:
                self.save_checkpoint(self.agents[ticker], ticker)
            self.logger.log(f"Error occurred: {e}", level="CRITICAL")
            raise e


if __name__ == "__main__":
    dm = NEATPacaTrader(TICKERS=["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD", "AVAX/USD"],
                      AGENT_GENERATIONS={"BTC/USD": 560,
                                         "ETH/USD": 400,
                                         "LTC/USD": 190,
                                         "DOGE/USD": 420,
                                         "AVAX/USD": 400}, 
                      TFRAMES=['30Min'],
                      API_KEY='PK50JGQU0JLZIJE9OM7A', 
                      API_SECRET='F8QdVDCLEjsIoXwhgsTQQW5kIQTogiZA8rDGfu4N')
    
    dm.trade()


