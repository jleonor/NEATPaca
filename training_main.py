import sys
from NEATPaca.DataHandler import DataHandler
from NEATPaca.Trainer import Trainer
from NEATPaca.ConfigReader import *
from multiprocessing import Process

def train_agent(ticker, config):
    # Initialise DataHandler
    handler = DataHandler(TICKER=ticker, 
                          config=config)
    handler.get_data(config=config)

    # Initialise Trainer
    trainer = Trainer(TICKER=ticker, 
                      config=config)
    trainer.run()


def start_training_process(ticker, config):
    p = Process(target=train_agent, args=(ticker, config))
    p.start()
    return p


if __name__ == "__main__":
    config = ConfigReader('config.config')
    processes = []

    if not config.threading:
        for ticker in config.tickers:
            train_agent(ticker, config)
    
    if config.threading:
        # Run each ticker training in a separate process
        for ticker in config.tickers:
            p = start_training_process(ticker, config)
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()