import numpy as np
import neat
import cupy as cp


class TradingAgent:
    def __init__(self, starting_money, transaction_fee_percent, config):
        self.starting_money = starting_money
        self.money = starting_money
        self.transaction_fee_percent = transaction_fee_percent
        self.position_open = False
        self.entry_price = 0
        self.config = config
        self.net = None
        self.trade_log = []
        self.daily_balances = [starting_money]
        self.wins = 0
        self.losses = 0
        self.equity_peak = starting_money
        self.max_drawdown = 0
        self.portfolio = starting_money
        self.last_action_period = 0

    def reset(self):
        self.money = self.starting_money
        self.position_open = False
        self.entry_price = 0
        self.trade_log = []
        self.daily_balances = [self.starting_money]
        self.portfolio = self.starting_money
        self.wins = 0
        self.losses = 0
        self.equity_peak = self.starting_money
        self.max_drawdown = 0
        self.last_action_period = 0

    def set_network(self, network_or_genome):
        if isinstance(network_or_genome, neat.DefaultGenome):
            # If it's a genome, create the network
            self.net = neat.nn.FeedForwardNetwork.create(network_or_genome, self.config)
        elif isinstance(network_or_genome, neat.nn.feed_forward.FeedForwardNetwork):
            # If it's already a network, just set it
            self.net = network_or_genome
        else:
            raise ValueError("Invalid network or genome provided")

    def make_decision(self, input_data, current_price):
        """Process the input data through the neural network to make a decision.
        Now includes position status as an additional input feature and potential percentage gain."""
        
        # Ensure input_data is a CuPy array
        input_data = cp.asarray(input_data)
        
        # Calculate potential percentage gain if position is open
        if self.position_open:
            sell_value = self.investment * (current_price / self.entry_price)  # Calculate sell value based on current price and investment
            transaction_fee = sell_value * self.transaction_fee_percent  # Transaction fee for selling
            sell_value -= transaction_fee  # Deduct transaction fee from sell value
            profit = sell_value - self.investment  # Calculate profit from the sell
            potential_gain = (profit / self.investment)  # Calculate percentage gain from the trade
        else:
            potential_gain = 0
        
        # Binary feature for position status (1 for open, 0 for not open)
        position_status = 1 if self.position_open else 0
        self.last_action_period += 1
        last_action_period = self.last_action_period / 1000
        
        # Add the position status and potential gain as features
        extended_input = cp.append(input_data, [potential_gain, position_status, last_action_period])
        output = self.net.activate(extended_input)  # Ensure that 'activate' function can handle CuPy arrays
        
        if isinstance(output, list):
            # Ensure all elements in the list are CuPy arrays
            output = cp.array(output)
        # print(output)
        # for i in output:
        #     print(i)
        return cp.argmax(output)

    def execute_trade(self, decision, price, date_time):
        """Execute a trade based on the decision and updates the agent's balance and position status, including the
        Date_Time of the trade."""
        if decision == 0 and not self.position_open:  # Buy decision
            transaction_fee = self.money * self.transaction_fee_percent  # Transaction fee for buying
            self.position_open = True
            self.entry_price = price
            units_bought = (self.money) / price  # Calculate units that can be bought after transaction fee
            self.money -= ((units_bought * price))  # Update money after buying units and paying fee
            self.investment = units_bought * price  # Record the amount invested in asset
            self.portfolio = self.investment * (price / self.entry_price)
            self.last_action_period = 0
            # Log the buy trade
            self.trade_log.append({'action': 'Buy', 'price': price, 'Date_Time': date_time, 'Balance': self.money,
                                   'Portfolio': self.portfolio})

        elif decision == 2 and self.position_open:  # Sell decision
            sell_value = self.investment * (
                        price / self.entry_price)  # Calculate sell value based on current price and investment
            transaction_fee = sell_value * self.transaction_fee_percent  # Transaction fee for selling
            sell_value -= transaction_fee  # Deduct transaction fee from sell value
            profit = sell_value - self.investment  # Calculate profit from the sell
            
            # Update wins or losses based on the trade outcome
            if profit > 0:
                self.wins += 1
            elif profit < 0:
                self.losses += 1

            self.money += sell_value  # Update money with the sell value
            self.position_open = False  # Close the position after selling
            percentage_gain = (profit / self.investment) * 100  # Calculate percentage gain from the trade
            self.portfolio = self.money
            self.last_action_period = 0

            # Log the sell trade with profit and percentage gain
            self.trade_log.append(
                {'action': 'Sell', 'price': price, 'Date_Time': date_time, 'Balance': self.money, 'Profit': profit,
                 'Percentage Gain': percentage_gain, 'Portfolio': self.portfolio})

            # Update equity peak and drawdown after selling
            self.equity_peak = max(self.equity_peak, self.money)
            current_drawdown = (self.equity_peak - self.money) / self.equity_peak
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def evaluate_on_data(self, df_data, df_ta_data):
        self.reset()  # Reset the agent for a new evaluation
        for i, row in df_ta_data.iterrows():
            input_data = row.to_numpy()
            current_price = df_data.iloc[i]['Close']
            decision = self.make_decision(input_data, current_price)
            date_time = df_data.iloc[i]['Date_Time']
            self.execute_trade(decision, current_price, date_time)

            # Assuming each row represents a new "day" or period for simplicity
            self.daily_balances.append(self.portfolio)

        # Calculate PnL as the difference between the final balance and the starting balance
        pnl = self.portfolio - self.starting_money
        return pnl
    
    def evaluate_on_row(self, row_ta, row_data):
        input_data = row_ta.to_numpy()
        current_price = row_data['Close']
        decision = self.make_decision(input_data, current_price)
        date_time = row_data['Date_Time']
        self.execute_trade(decision, current_price, date_time)
        self.daily_balances.append(self.portfolio)
        pnl = self.portfolio - self.starting_money
        return pnl

    def log_trades(self):
        """Print the log of all trades."""
        for trade in self.trade_log:
            print(trade)
