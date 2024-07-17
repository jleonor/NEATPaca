from ConfigReader import ConfigReader
import subprocess

config = ConfigReader('config.config')

TICKERS = config.tickers
TFRAMES = ",".join(config.timeframes)  # Since we're passing this through command line, join it into a string
from_year = config.from_year
from_month = config.from_month
from_day = config.from_day

if __name__ == '__main__':
    for ticker in TICKERS:
        # Construct the command to run in a new terminal
        command = f"python train_main.py {ticker} {from_year} {from_month} {from_day} {TFRAMES}"
        # Use subprocess to open a new terminal for each process
        subprocess.run(command, shell=True)