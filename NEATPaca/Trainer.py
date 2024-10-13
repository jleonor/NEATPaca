import pandas as pd
import neat
from NEATPaca.NEATAgent import *
from NEATPaca.Logger import *
from NEATPaca.ConfigReader import *
# import pickle
import dill as pickle
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import json


class Trainer():
    def __init__(self, TICKER, config, config_path='neat_config.txt', checkpoint_load=''):
        self.logger = Logger(folder_path="train_logs", 
                        file_name="train_log",
                        level="DEBUG",
                        rotation="daily",
                        archive_freq="daily")
        
        self.ticker = TICKER
        self.TFRAMES = config.timeframes
        self.from_year = config.from_year
        self.from_month = config.from_month
        self.from_day = config.from_day

        today = datetime.today()
        self.until_year = config.until_year if config.until_year is not None else today.year
        self.until_month = config.until_month if config.until_month is not None else today.month
        self.until_day = config.until_day if config.until_day is not None else today.day


        self.config_path = config_path

        self.checkpoint_load = checkpoint_load
        self.starting_money = config.starting_money
        self.transaction_fee_percent = config.transaction_fee
        self.name = self.create_filename()
        self.model_folder_name = "./models"
        # self.viz_folder_name = "./viz_outputs"
        self.checkpoint_folder_name = "./training_checkpoints"
        self.training_results = "./training_results"

        self.create_folders()
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  self.config_path)
        self.load_population()
        self.load_data()

    def create_folders(self):
        """Creates necessary folders if they do not exist."""
        os.makedirs(self.model_folder_name, exist_ok=True)
        # os.makedirs(self.viz_folder_name, exist_ok=True)
        os.makedirs(self.checkpoint_folder_name, exist_ok=True)
        os.makedirs(self.training_results, exist_ok=True)

    def add_reporters(self):
        """Add NEAT reporters for output and saving checkpoints."""
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())
        self.population.add_reporter(CustomCheckpointer(generation_interval=1, filename_prefix=f'{self.checkpoint_folder_name}/{self.name}_checkpoint_'))
        self.population.add_reporter(CustomEvaluationReporter(self))

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint file in the checkpoint directory."""
        checkpoints = [os.path.join(self.checkpoint_folder_name, f) for f in os.listdir(self.checkpoint_folder_name) if f.startswith(self.name + '_checkpoint_')]
        if checkpoints:
            return max(checkpoints, key=os.path.getctime)
        return None

    def create_filename(self):
        ticker_sanitized = self.ticker.replace("/", "-")
        tframes_joined = '-'.join(self.TFRAMES)
        self.name = f"{ticker_sanitized}_{tframes_joined}_{self.from_year}_{self.from_month}_{self.from_day}__{self.until_year}_{self.until_month}_{self.until_day}"

        return self.name

    def update_config(self, expected_inputs):
        import configparser
        config = configparser.ConfigParser()
        config.read(self.config_path)
        config.set('DefaultGenome', 'num_inputs', str(expected_inputs))
        with open(self.config_path, 'w') as configfile:
            config.write(configfile)
        print(f"Updated configuration: num_inputs set to {expected_inputs}")


    def load_population(self):
        if self.checkpoint_load:
            self.population = CustomCheckpointer.restore_checkpoint(self.checkpoint_load)
        else:
            self.population = neat.Population(self.config)

        self.add_reporters()

    def load_data(self):
        file_path = f'price_data/{self.name}.csv'
        df = pd.read_csv(file_path)
        df.ffill(inplace=True)

        # Define columns
        cols_to_remove = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        cols = df.columns.tolist()
        ta_columns = [col for col in cols if col not in cols_to_remove]
        df_ta = df[ta_columns]

        train_size = int(len(df) * 0.9)
        self.df_train = df.iloc[:train_size]
        self.df_ta_train = df_ta.iloc[:train_size]
        self.df_ta_test = df_ta.iloc[train_size:].reset_index(drop=True)
        self.df_test = df.iloc[train_size:].reset_index(drop=True)

    def select_random_subset(self):
        subset_size = int(0.02 * len(self.df_train))  # 2% of the total training dataset
        minimum_rows = 200  # Minimum threshold for rows

        if subset_size < minimum_rows:
            error_message = f"Subset size ({subset_size}) is below the minimum threshold ({minimum_rows})"
            self.logger.log(error_message, level="CRITICAL")
            raise ValueError(error_message)

        # Randomly select a starting index for the subset
        start_idx = random.randint(0, len(self.df_train) - subset_size)
        end_idx = start_idx + subset_size

        # Create the subset for this generation
        self.subset_df_train = self.df_train.iloc[start_idx:end_idx].reset_index(drop=True)
        self.subset_df_ta_train = self.df_ta_train.iloc[start_idx:end_idx].reset_index(drop=True)
        self.logger.log(f"Selected new data subset for generation: rows {start_idx} to {end_idx}", level="DEBUG")


    def eval_genomes(self, genomes, config):
        self.select_random_subset()  # Only select a subset for training

        for genome_id, genome in genomes:
            agent = TradingAgent(self.starting_money, self.transaction_fee_percent, config)
            agent.set_network(genome)
            # Use the subset data for evaluation
            agent.evaluate_on_data(self.subset_df_train, self.subset_df_ta_train)
            self.calculate_fitness(genome, agent)


    def calculate_fitness(self, genome, agent):
        num_trades = len(agent.trade_log)
        if num_trades < 2:
            genome.fitness = -200
        else:
            win_loss_ratio, consistency_bonus, sum_percentage_gains = self.calculate_metrics(agent)
            trade_candle_ratio = len(agent.trade_log)/len(self.df_train)
            fitness = (((win_loss_ratio - 1) + (consistency_bonus/2000)) * (trade_candle_ratio * 12)) * sum_percentage_gains
            genome.fitness = -abs(fitness) if win_loss_ratio < 0 or sum_percentage_gains < 0 else abs(fitness)

    def calculate_metrics(self, agent):
        # Calculate win/loss ratio
        win_loss_ratio = agent.wins / max(1, agent.losses)  # Avoid division by zero

        # Check for invalid daily balance elements and calculate daily returns
        for i, balance in enumerate(agent.daily_balances):
            if not isinstance(balance, (int, float)):
                print(f"Problematic element at index {i}: {balance} (Type: {type(balance)})")
                return None  # Or handle error as appropriate

        daily_returns = np.diff(agent.daily_balances) / agent.daily_balances[:-1]
        std_daily_returns = np.std(daily_returns)
        
        # Calculate consistency bonus
        consistency_bonus = 1 / std_daily_returns if std_daily_returns > 0 else 0

        # Initialize variables for sum of percentage gains
        sum_percentage_gains = 0
        for trade in agent.trade_log:
            if 'Percentage Gain' in trade:
                # Sum of percentage gains
                sum_percentage_gains += trade['Percentage Gain']

        return win_loss_ratio, consistency_bonus, sum_percentage_gains

    def run(self, attempt=1, max_attempts=2):
        try:
            self.create_folders()
            winner = self.population.run(self.eval_genomes, 1000)
            train_winner_agent = TradingAgent(self.starting_money, self.transaction_fee_percent, self.config)
            train_winner_agent.set_network(winner)
            train_winner_agent.reset()

            for i in range(len(self.df_train)):
                row_ta = self.df_ta_train.iloc[i]
                row_data = self.df_train.iloc[i]
                train_winner_agent.evaluate_on_row(row_ta, row_data)

            train_pnl = train_winner_agent.portfolio - train_winner_agent.starting_money
            train_market_pnl = self.calculate_market_performance(self.df_train)

            winner_agent = TradingAgent(self.starting_money, self.transaction_fee_percent, self.config)
            winner_agent.set_network(winner)
            winner_agent.reset()

            for i in range(len(self.df_test)):
                row_ta = self.df_ta_test.iloc[i]
                row_data = self.df_test.iloc[i]
                winner_agent.evaluate_on_row(row_ta, row_data)

            test_pnl = winner_agent.portfolio - winner_agent.starting_money
            test_market_pnl = self.calculate_market_performance(self.df_test)

            try:
                win_loss_ratio, consistency_bonus, sum_percentage_gains = self.calculate_metrics(winner_agent)
                test_metrics = [round(win_loss_ratio - 1, 4), round(consistency_bonus / 2000, 4), round(sum_percentage_gains, 4), round(test_pnl, 2), round(test_market_pnl, 2), len(winner_agent.trade_log), round((len(winner_agent.trade_log) / len(self.df_test)) * 12, 4)]
                win_loss_ratio, consistency_bonus, sum_percentage_gains = self.calculate_metrics(train_winner_agent)
                train_metrics = [round(win_loss_ratio - 1, 4), round(consistency_bonus / 2000, 4), round(sum_percentage_gains, 4), round(train_pnl, 2), round(train_market_pnl, 2), len(train_winner_agent.trade_log), round((len(train_winner_agent.trade_log) / len(self.df_train)) * 12, 4)]
                # self.visualize_trades_with_plotly(self.df_test, winner_agent.trade_log, self.df_train, train_winner_agent.trade_log, test_metrics, train_metrics, gen='WINNER')
            except Exception as e:
                print(f"An error occurred: {e}")

            with open(f'models/winner_{self.name}.pkl', 'wb') as f:
                pickle.dump(winner, f, protocol=pickle.HIGHEST_PROTOCOL)

        except RuntimeError as e:
            error_message = str(e)
            if "Expected" in error_message and "inputs, got" in error_message and attempt < max_attempts:
                expected, got = map(int, error_message.split("Expected ")[1].split(" inputs, got "))
                if got != expected:
                    print(f"Error encountered: {error_message}")
                    print(f"Attempting to update configuration to set expected inputs from {expected} to {got} and rerunning...")
                    self.update_config(got)
                    self.reload_configuration()  # Method to reload the configuration
                    self.run(attempt + 1)  # Recursive call to try again
                else:
                    raise e  # Re-raise the exception if the number of inputs provided is more than expected
            else:
                # When the max attempts are exceeded or the error is not related to input count
                if attempt >= max_attempts:
                    raise Exception(f"Maximum attempts exceeded. Failed to resolve the issue after {max_attempts} attempts.")
                else:
                    raise e  # Re-raise the original exception if it's a different kind of RuntimeError
        
        # except Exception as e:
        #     raise Exception(f"An unexpected error occurred: {str(e)}")

    def reload_configuration(self):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                self.config_path)
        self.load_population()  # Reload the population with the updated configuration

    # Utility function to format metric names
    def format_metric_name(self, name):
        return ' '.join(word.capitalize() for word in name.split('_'))

    # Data Preparation: Align trade log entries with DataFrame dates
    def align_trade_log(self, trade_log, df):
        if trade_log[0]['Date_Time'] != df['Date_Time'].iloc[0]:
            first_trade_log_entry = trade_log[0].copy()
            first_trade_log_entry['Date_Time'] = df['Date_Time'].iloc[0]
            first_trade_log_entry['action'] = 'N/A'
            trade_log.insert(0, first_trade_log_entry)

        if trade_log[-1]['Date_Time'] != df['Date_Time'].iloc[-1]:
            last_trade_log_entry = trade_log[-1].copy()
            last_trade_log_entry['Date_Time'] = df['Date_Time'].iloc[-1]
            last_trade_log_entry['action'] = 'N/A'
            trade_log.append(last_trade_log_entry)

        return pd.DataFrame(trade_log)

    def prepare_viz_data(self, df_train, trade_log_train, df_test, trade_log_test):
        trade_log_train_df = self.align_trade_log(trade_log_train, self.df_train)
        trade_log_test_df = self.align_trade_log(trade_log_test, self.df_test)
        return trade_log_train_df, trade_log_test_df

    # Visualization functions: Add different types of plots to the figure
    def add_price_chart(self, fig, df, trade_log_df, row, col):
        fig.add_trace(go.Scatter(x=df['Date_Time'], y=df['Close'], mode='lines', name='Close Price'), row=row, col=col)
        buy_signals = trade_log_df[trade_log_df['action'] == 'Buy']
        sell_signals = trade_log_df[trade_log_df['action'] == 'Sell']
        fig.add_trace(go.Scatter(x=buy_signals['Date_Time'], y=buy_signals['price'], mode='markers', name='Buy', 
                                marker=dict(color='green', size=10, symbol='triangle-up')), row=row, col=col)
        fig.add_trace(go.Scatter(x=sell_signals['Date_Time'], y=sell_signals['price'], mode='markers', name='Sell', 
                                marker=dict(color='red', size=10, symbol='triangle-down')), row=row, col=col)

    def add_metrics_table(self, fig, metrics, col, row=1):
        headers = [self.format_metric_name(name) for name in ['Win loss ratio', 'Consistency bonus', 'Sum percentage gains', 'Agent PnL', 'Market PnL', 'Number of Trades', 'Trade to Candle Ratio']]
        fig.add_trace(
            go.Table(
                header=dict(values=headers, fill_color='grey', align='left', font=dict(color='white', size=12)),
                cells=dict(values=np.reshape(metrics, (7, 1)), align='left')),
            row=row, col=col
        )

    def add_agent_money_chart(self, fig, trade_log_df, row, col):
        fig.add_trace(
            go.Scatter(
                x=trade_log_df['Date_Time'],
                y=trade_log_df['Portfolio'],
                mode='lines',
                name='Agent Money',
                showlegend=False
            ),
            row=row, col=col
        )

    def add_trade_history_table(self, fig, trade_log_df, row, col):
        valid_trades = trade_log_df[trade_log_df['action'] != 'N/A']
        table_data = valid_trades[['Date_Time', 'action', 'price', 'Portfolio']]
        table_data.loc[:, 'Date_Time'] = table_data['Date_Time'].astype(str).str[:-9]
        colors = [['#ABEBC6' if action == 'Buy' else '#F5B7B1' for action in table_data['action']]]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Date_Time', 'Action', 'Price', 'Portfolio'],
                    fill_color='grey',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[table_data.Date_Time, table_data.action, table_data.price, table_data.Portfolio],
                    fill_color=colors,
                    align='left'
                )
            ),
            row=row, col=col
        )


    # Main function to visualize trades with Plotly
    def visualize_trades_with_plotly(self, df_test, trade_log_test, df_train, trade_log_train, test_metrics, train_metrics, gen=''):
        trade_log_train_df, trade_log_test_df = self.prepare_viz_data(self.df_train, trade_log_train, self.df_test, trade_log_test)
        fig = self.create_figure(self.df_train, self.df_test)
        
        # Add price charts for train and test data
        self.add_price_chart(fig, self.df_train, trade_log_train_df, row=2, col=1)
        self.add_price_chart(fig, self.df_test, trade_log_test_df, row=2, col=2)
        
        # Add metrics tables for train and test data
        self.add_metrics_table(fig, train_metrics, col=1)
        self.add_metrics_table(fig, test_metrics, col=2)
        
        # Add agent money charts for train and test data
        self.add_agent_money_chart(fig, trade_log_train_df, row=3, col=1)
        self.add_agent_money_chart(fig, trade_log_test_df, row=3, col=2)
        
        # Add trade history tables for train and test data
        self.add_trade_history_table(fig, trade_log_train_df, row=4, col=1)
        self.add_trade_history_table(fig, trade_log_test_df, row=4, col=2)
        
        fig.update_layout(
            title={
                'text': 'Trade History, Agent Money Over Time, and Trade Log (Generation #' + str(gen) + ')',
                'x': 0.5,
                'xanchor': 'center'
            }
        )
        
        # Save the figure as an HTML file
        html_file_path = os.path.join('viz_outputs', self.name + '_Gen_' + str(gen) + '.html')
        fig.write_html(html_file_path)


    # Helper to create the main figure layout
    def create_figure(self, df_train, df_test):
        return make_subplots(rows=4, cols=2, shared_xaxes=True, vertical_spacing=0.04, horizontal_spacing=0.05,
                            row_heights=[0.1, 0.30, 0.30, 0.30],
                            subplot_titles=(f'Agent on TRAIN data | {str(df_train["Date_Time"].iloc[0])[:-9]} to {str(df_train["Date_Time"].iloc[-1])[:-9]}',
                                            f'Agent on TEST data | {str(df_test["Date_Time"].iloc[0])[:-9]} to {str(df_test["Date_Time"].iloc[-1])[:-9]}', 
                                            '',
                                            '',
                                            '',
                                            '',
                                            '',
                                            ''),
                            specs=[[{"type": "table"}, {"type": "table"}],
                                    [{"type": "scatter"}, {"type": "scatter"}],
                                    [{"type": "scatter"}, {"type": "scatter"}],
                                    [{"type": "table"}, {"type": "table"}]])

    # Function to evaluate the winner on the test data
    def evaluate_winner_on_test_data(self, winner, config):
        winner_agent = TradingAgent(self.starting_money, self.transaction_fee_percent, config)
        winner_agent.set_network(winner)
        winner_agent.reset()  # Reset the agent before evaluating on the test set
        test_pnl = winner_agent.evaluate_on_data(self.df_test, self.df_ta_test)

        return winner_agent, test_pnl


    # Function to evaluate the winner on the train data
    def evaluate_winner_on_train_data(self, winner, config):
        train_winner_agent = TradingAgent(self.starting_money, self.transaction_fee_percent, config)
        train_winner_agent.set_network(winner)
        train_winner_agent.reset()  # Reset before training evaluation
        train_pnl = train_winner_agent.evaluate_on_data(self.df_train, self.df_ta_train)
        return train_winner_agent, train_pnl

    # Market Performance (Buy-and-Hold Strategy)
    def calculate_market_performance(self, df_market):
        initial_price = df_market.iloc[0]['Close']
        final_price = df_market.iloc[-1]['Close']
        bitcoins_bought = self.starting_money / initial_price
        final_value = bitcoins_bought * final_price
        market_pnl = final_value - self.starting_money
        return market_pnl
    
    def save_generation_data(self, gen, metrics, train_trade_log, test_trade_log):
        parent_folder = os.path.join('training_results', self.name)
        generation_folder = os.path.join(parent_folder, f"_Gen_{gen}")
        os.makedirs(generation_folder, exist_ok=True)

        # Save metrics as JSON
        with open(os.path.join(generation_folder, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)

        # Ensure each trade log reflects the correct evaluation
        train_trade_log_df = pd.DataFrame(train_trade_log)
        test_trade_log_df = pd.DataFrame(test_trade_log)

        # Save separate CSV files for train and test logs
        train_trade_log_df.to_csv(os.path.join(generation_folder, 'train_trade_log.csv'), index=False)
        test_trade_log_df.to_csv(os.path.join(generation_folder, 'test_trade_log.csv'), index=False)



class CustomEvaluationReporter(neat.reporting.BaseReporter):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer  # Store a reference to the Trainer instance
        self.generation = 0  # Initialize a generation counter

    def start_generation(self, generation):
        self.generation = generation
        self.trainer.logger.log(f"Running generation {generation} for {self.trainer.ticker}", level="INFO")
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        evaluated_genomes = [g for g in population.values() if g.fitness is not None]
        
        if evaluated_genomes:
            best_genome = max(evaluated_genomes, key=lambda g: g.fitness)

            # Train evaluation on the subset (already handled in eval_genomes)
            train_winner_agent, train_pnl = self.trainer.evaluate_winner_on_train_data(best_genome, config)
            train_market_pnl = self.trainer.calculate_market_performance(self.trainer.df_train)
            
            # Full Test Set Evaluation
            winner_agent, test_pnl = self.trainer.evaluate_winner_on_test_data(best_genome, config)
            test_market_pnl = self.trainer.calculate_market_performance(self.trainer.df_test)

            # Calculate metrics for the test data on the full dataset
            win_loss_ratio, consistency_bonus, sum_percentage_gains = self.trainer.calculate_metrics(winner_agent)
            train_metrics = {
                "win_loss_ratio": round(win_loss_ratio - 1, 4),
                "consistency_bonus": round(consistency_bonus / 2000, 4),
                "sum_percentage_gains": round(sum_percentage_gains, 4),
                "test_pnl": round(test_pnl, 2),
                "test_market_pnl": round(test_market_pnl, 2),
                "num_trades": len(winner_agent.trade_log),
                "trade_to_candle_ratio": round((len(winner_agent.trade_log) / len(self.trainer.df_test)) * 12, 4)
            }

            # Calculate metrics for the train data on the subset
            win_loss_ratio, consistency_bonus, sum_percentage_gains = self.trainer.calculate_metrics(train_winner_agent)
            test_metrics = {
                "win_loss_ratio": round(win_loss_ratio - 1, 4),
                "consistency_bonus": round(consistency_bonus / 2000, 4),
                "sum_percentage_gains": round(sum_percentage_gains, 4),
                "train_pnl": round(train_pnl, 2),
                "train_market_pnl": round(train_market_pnl, 2),
                "num_trades": len(train_winner_agent.trade_log),
                "trade_to_candle_ratio": round((len(train_winner_agent.trade_log) / len(self.trainer.df_train)) * 12, 4)
            }
            
            # Save generation data including both train and test trade logs
            self.trainer.save_generation_data(
                gen=self.generation,
                metrics={"train": train_metrics, "test": test_metrics},
                train_trade_log=winner_agent.trade_log,
                test_trade_log=train_winner_agent.trade_log,
            )
            
            self.trainer.logger.log(f"Generation {self.generation} data saved.", level="INFO")
            self.generation += 1
        else:
            print("No genomes were evaluated in this generation.")

class CustomCheckpointer(neat.Checkpointer):
    def save_checkpoint(self, config, population, species_set, generation):
        # Remove lock or other threading dependencies if necessary here
        self.clean_population(population)
        # Then save as usual
        super().save_checkpoint(config, population, species_set, generation)

    def clean_population(self, population):
        for genome in population.values():
            if hasattr(genome, 'lock'):
                del genome.lock  # Or any similar lock object


if __name__ == "__main__":
    config = ConfigReader('config.config')
    trainer = Trainer(TICKER="ETH/USD",
                      config=config,
                      TFRAMES=['30Min', '2H'],
                      from_year=2023, 
                      from_month=10, 
                      from_day=1)
    trainer.run()
