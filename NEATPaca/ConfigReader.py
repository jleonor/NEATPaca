import configparser

class ConfigReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tickers = []
        self.agent_generations = {}
        self.timeframes = []
        self.api_key = None
        self.api_secret = None
        self.from_year = None
        self.from_month = None
        self.from_day = None
        self.load_config()
        
    def load_config(self):
        # Create a config parser object
        config = configparser.ConfigParser()
        # Read the configuration file
        config.read(self.filepath)
        
        # Accessing the data from the config
        self.tickers = config['general']['TICKERS'].split(', ')
        self.timeframes = config['general']['TIMEFRAMES'].split(', ')
        from_date = config['training']['FROM_DATE'].split('-')
        self.api_key = config['trading']['API_KEY']
        self.api_secret = config['trading']['API_SECRET']
        agent_gens = list(map(int, config['trading']['AGENT_GENS'].split(', ')))

        # Map tickers to their respective generations
        self.agent_generations = {ticker: gen for ticker, gen in zip(self.tickers, agent_gens)}

        # Convert from_date to integers
        self.from_year, self.from_month, self.from_day = map(int, from_date)

if __name__ == "__main__":
    # Load the config
    filepath = 'config.config'
    config = ConfigReader(filepath)
    
    print("Tickers:", config.tickers)
    print("Timeframes:", config.timeframes)
    print("Agent Generations:", config.agent_generations)
    print("API Key:", config.api_key)
    print("API Secret:", config.api_secret)
    print("From Year:", config.from_year)
    print("From Month:", config.from_month)
    print("From Day:", config.from_day)
