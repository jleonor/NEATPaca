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
        self.until_year = None
        self.until_month = None
        self.until_day = None
    
        self.threading = None
        self.starting_money = None
        self.transaction_fee = None

        self.use_ma_cross = None
        self.use_ma_conv_div = None
        self.use_ma_slope = None
        self.ma_conv_div_periods = None
        self.ma_cross_periods = None
        self.use_bbands = None
        self.bbands_periods = None
        self.use_stoch_cross = None
        self.stoch_periods = None
        self.use_macd_cross = None
        self.macd_cross_periods = None
        self.use_obv_ma_cross = None
        self.obv_period = None
        self.use_pct_change = None
        self.pct_change_period = None
        self.use_psar_cl_cross = None
        self.use_ultosc = None
        self.ultosc_1_period = None
        self.ultosc_2_period = None

        self.use_overwrite = None
        self.overwrite_values = None
        self.use_adx = None
        self.adx_periods = None
        self.use_rsi = None
        self.rsi_periods = None
        self.use_cci = None
        self.cci_periods = None
        self.use_mfi = None
        self.mfi_periods = None
        self.use_aroonosc = None
        self.aroonosc_periods = None
        self.use_cmo = None
        self.cmo_periods = None
        self.use_hi_lo_pct = None
        self.hi_lo_pct_periods = None

        self.load_config()
        if self.use_overwrite:
            self.overwrite_indicators()
        
    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.filepath)
        
        # General Parsing Function
        def parse_value(section, key, default=None):
            try:
                # print(f"{key}: {config[section][key].split(';')[0].strip()}")
                value = config[section][key].split(';')[0].strip()
                return value
            except KeyError:
                return default

        # Load General and Trading Configurations
        self.tickers = parse_value('GENERAL', 'tickers').split(', ')
        self.timeframes = parse_value('GENERAL', 'timeframes').split(', ')
        self.api_key = parse_value('TRADING', 'api_key')
        self.api_secret = parse_value('TRADING', 'api_secret')
        agent_gens = list(map(int, parse_value('TRADING', 'agent_gens').split(', ')))

        # Load Training Dates
        from_date = parse_value('TRAINING', 'from_date').split('-')
        until_date = parse_value('TRAINING', 'until_date')
        until_date = False if until_date.lower() == "false" else until_date.split('-')

        # Map tickers to their respective generations
        self.agent_generations = dict(zip(self.tickers, agent_gens))
        
        # Convert date components to integers
        self.from_day, self.from_month, self.from_year = map(int, from_date)
        if until_date:
            self.until_day, self.until_month, self.until_year = map(int, until_date)

        # Threading and Money Configurations
        self.threading = parse_value('TRAINING', 'threading') == 'TRUE'
        self.starting_money = int(parse_value('TRAINING', 'starting_money'))
        self.transaction_fee = float(parse_value('TRAINING', 'transaction_fee'))

        # Indicator Configurations
        self.use_ma_cross = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_ma_cross') == 'TRUE'
        self.use_ma_conv_div = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_ma_conv_div') == 'TRUE'
        self.use_ma_slope = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_ma_slope') == 'TRUE'

        self.ma_conv_div_periods = list(map(int, parse_value('INDICATORS_NOT_OVERWRITABLE', 'ma_conv_div_periods').split(', ')))
        self.ma_cross_periods = list(map(int, parse_value('INDICATORS_NOT_OVERWRITABLE', 'ma_cross_periods').split(', ')))

        self.use_bbands = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_bbands') == 'TRUE'
        bbands_settings = parse_value('INDICATORS_NOT_OVERWRITABLE', 'bbands_periods').split(', ')
        self.bbands_periods = list(map(int, bbands_settings[:1])) + list(map(float, bbands_settings[1:]))

        self.use_stoch_cross = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_stoch_cross') == 'TRUE'
        self.stoch_periods = list(map(int, parse_value('INDICATORS_NOT_OVERWRITABLE', 'stoch_periods').split(', ')))

        self.use_macd_cross = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_macd_cross') == 'TRUE'
        self.macd_cross_periods = list(map(int, parse_value('INDICATORS_NOT_OVERWRITABLE', 'macd_cross_periods').split(', ')))

        self.use_obv_ma_cross = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_obv_ma_cross') == 'TRUE'
        self.obv_ma_period = int(parse_value('INDICATORS_NOT_OVERWRITABLE', 'obv_ma_period'))

        self.use_pct_change = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_pct_change') == 'TRUE'
        self.pct_change_period = int(parse_value('INDICATORS_NOT_OVERWRITABLE', 'pct_change_period'))

        self.use_psar_cl_cross = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_psar_cl_cross') == 'TRUE'

        self.use_ultosc = parse_value('INDICATORS_NOT_OVERWRITABLE', 'use_ultosc') == 'TRUE'
        ultosc_settings = parse_value('INDICATORS_NOT_OVERWRITABLE', 'ultosc_1_periods').split(', ')
        self.ultosc_1_periods = list(map(int, ultosc_settings))
        ultosc_settings = parse_value('INDICATORS_NOT_OVERWRITABLE', 'ultosc_2_periods').split(', ')
        self.ultosc_2_periods = list(map(int, ultosc_settings))

        # Overwritable Indicator Configurations
        self.use_overwrite = parse_value('INDICATORS_OVERWRITABLE', 'use_overwrite') == 'TRUE'
        self.overwrite_values = list(map(int, parse_value('INDICATORS_OVERWRITABLE', 'overwrite_values').split(', ')))

        self.use_adx = parse_value('INDICATORS_OVERWRITABLE', 'use_adx') == 'TRUE'
        self.adx_periods = int(parse_value('INDICATORS_OVERWRITABLE', 'adx_periods')) if parse_value('INDICATORS_OVERWRITABLE', 'adx_periods') != 'FALSE' else False
        
        self.use_rsi = parse_value('INDICATORS_OVERWRITABLE', 'use_rsi') == 'TRUE'
        self.rsi_periods = int(parse_value('INDICATORS_OVERWRITABLE', 'rsi_periods')) if parse_value('INDICATORS_OVERWRITABLE', 'rsi_periods') != 'FALSE' else False
        
        self.use_cci = parse_value('INDICATORS_OVERWRITABLE', 'use_cci') == 'TRUE'
        self.cci_periods = int(parse_value('INDICATORS_OVERWRITABLE', 'cci_periods')) if parse_value('INDICATORS_OVERWRITABLE', 'cci_periods') != 'FALSE' else False
        
        self.use_mfi = parse_value('INDICATORS_OVERWRITABLE', 'use_mfi') == 'TRUE'
        self.mfi_periods = int(parse_value('INDICATORS_OVERWRITABLE', 'mfi_periods')) if parse_value('INDICATORS_OVERWRITABLE', 'mfi_periods') != 'FALSE' else False

        self.use_aroonosc = parse_value('INDICATORS_OVERWRITABLE', 'use_aroonosc') == 'TRUE'
        self.aroonosc_periods = int(parse_value('INDICATORS_OVERWRITABLE', 'aroonosc_periods')) if parse_value('INDICATORS_OVERWRITABLE', 'aroonosc_periods') != 'FALSE' else False

        self.use_cmo = parse_value('INDICATORS_OVERWRITABLE', 'use_cmo') == 'TRUE'
        self.cmo_periods = int(parse_value('INDICATORS_OVERWRITABLE', 'cmo_periods')) if parse_value('INDICATORS_OVERWRITABLE', 'cmo_periods') != 'FALSE' else False

        self.use_hi_lo_pct = parse_value('INDICATORS_OVERWRITABLE', 'use_hi_lo_pct') == 'TRUE'
        self.hi_lo_pct_periods = int(parse_value('INDICATORS_OVERWRITABLE', 'hi_lo_pct_periods')) if parse_value('INDICATORS_OVERWRITABLE', 'hi_lo_pct_periods') != 'FALSE' else False

    def overwrite_indicators(self):
        if self.use_adx and not self.adx_periods:
            self.adx_periods = self.overwrite_values

        if self.use_rsi and not self.rsi_periods:
            self.rsi_periods = self.overwrite_values

        if self.use_cci and not self.cci_periods:
            self.cci_periods = self.overwrite_values

        if self.use_mfi and not self.mfi_periods:
            self.mfi_periods = self.overwrite_values

        if self.use_aroonosc and not self.aroonosc_periods:
            self.aroonosc_periods = self.overwrite_values

        if self.use_cmo and not self.cmo_periods:
            self.cmo_periods = self.overwrite_values

        if self.use_hi_lo_pct and not self.hi_lo_pct_periods:
            self.hi_lo_pct_periods = self.overwrite_values

    def print_config(self):
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Tickers\t\t\t\t\t|\t", self.tickers)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Timeframes\t\t\t\t|\t", self.timeframes)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Agent Generations\t\t\t|\t", self.agent_generations)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("API Key\t\t\t\t\t|\t", self.api_key)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("API Secret\t\t\t\t|\t", self.api_secret)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Threading\t\t\t\t|\t", self.threading)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Starting Money\t\t\t\t|\t", self.starting_money)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Transaction Fee\t\t\t\t|\t", self.transaction_fee)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("From Year\t\t\t\t|\t", self.from_year)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("From Month\t\t\t\t|\t", self.from_month)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("From Day\t\t\t\t|\t", self.from_day)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Until Year\t\t\t\t|\t", self.until_year)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Until Month\t\t\t\t|\t", self.until_month)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Until Day\t\t\t\t|\t", self.until_day)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use Moving Average Cross\t\t|\t", self.use_ma_cross)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use MA Convergence Divergence\t\t|\t", self.use_ma_conv_div)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use MA Slope\t\t\t\t|\t", self.use_ma_slope )
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("MA Convergence Divergence Periods\t|\t", self.ma_conv_div_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("MA Cross Periods\t\t\t|\t", self.ma_cross_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use Bollinger Bands\t\t\t|\t", self.use_bbands)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Bollinger Bands Periods\t\t\t|\t", self.bbands_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use Stochastic Cross\t\t\t|\t", self.use_stoch_cross)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Stochastic Periods\t\t\t|\t", self.stoch_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use MACD Cross\t\t\t\t|\t", self.use_macd_cross)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("MACD Cross Periods\t\t\t|\t", self.macd_cross_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use OBV MA Cross\t\t\t|\t", self.use_obv_ma_cross)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("OBV Period\t\t\t\t|\t", self.obv_ma_period)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use Percentage Change\t\t\t|\t", self.use_pct_change)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Percentage Change Period\t\t|\t", self.pct_change_period)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use PSAR CL Cross\t\t\t|\t", self.use_psar_cl_cross)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use Ultimate Oscillator\t\t\t|\t", self.use_ultosc)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Ultimate Oscillator 1 Period\t\t|\t", self.ultosc_1_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Ultimate Oscillator 2 Period\t\t|\t", self.ultosc_2_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use Overwrite Values\t\t\t|\t", self.use_overwrite)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Overwrite Values\t\t\t|\t", self.overwrite_values)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use ADX\t\t\t\t\t|\t", self.use_adx)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("ADX Periods\t\t\t\t|\t", self.adx_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use RSI\t\t\t\t\t|\t", self.use_rsi)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("RSI Periods\t\t\t\t|\t", self.rsi_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use CCI\t\t\t\t\t|\t", self.use_cci)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("CCI Periods\t\t\t\t|\t", self.cci_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use MFI\t\t\t\t\t|\t", self.use_mfi)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("MFI Periods\t\t\t\t|\t", self.mfi_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use Aroon Oscillator\t\t\t|\t", self.use_aroonosc)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Aroon Oscillator Periods\t\t|\t", self.aroonosc_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use CMO\t\t\t\t\t|\t", self.use_cmo)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("CMO Periods\t\t\t\t|\t", self.cmo_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("Use High-Low Percentage\t\t\t|\t", self.use_hi_lo_pct)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
            print("High-Low Percentage Periods\t\t|\t", self.hi_lo_pct_periods)
            print("---------------------------------------------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    filepath = 'config.config'
    config = ConfigReader(filepath)
    config.print_config()