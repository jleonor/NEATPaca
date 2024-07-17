from NEATPacaTrader import *
from ConfigReader import *

config = ConfigReader('config.config')

dm = NEATPacaTrader(TICKERS=config.tickers,
                  AGENT_GENERATIONS=config.agent_generations,
                  TFRAMES=config.timeframes,
                  API_KEY=config.api_key,
                  API_SECRET=config.api_secret)
    
dm.trade()