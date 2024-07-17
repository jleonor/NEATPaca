import sys
from DataHandler import DataHandler
from Trainer import Trainer

def train_agent(ticker, from_year, from_month, from_day, timeframes):
    # Initialise DataHandler
    handler = DataHandler(TICKER=ticker, TFRAMES=timeframes, from_year=from_year, from_month=from_month, from_day=from_day)
    handler.get_data()

    # Initialise Trainer
    trainer = Trainer(TICKER=ticker, TFRAMES=timeframes, from_year=from_year, from_month=from_month, from_day=from_day)
    trainer.run()

if __name__ == "__main__":
    ticker = sys.argv[1]
    from_day = int(sys.argv[2])
    from_month = int(sys.argv[3])
    from_year = int(sys.argv[4])
    timeframes = sys.argv[5].split(',')

    print((ticker, from_year, from_month, from_day, timeframes))

    train_agent(ticker, from_year, from_month, from_day, timeframes)
