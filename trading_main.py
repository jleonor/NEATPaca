from NEATPaca.AlpacaTrader import *
from NEATPaca.ConfigReader import *

config = ConfigReader('config.config')
npt = NEATPacaTrader(config=config)

npt.trade()