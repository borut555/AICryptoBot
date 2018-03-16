import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import Imputer, scale
from sklearn.decomposition import PCA, FastICA
from hmmlearn import hmm
from poloniex_API import poloniex
from API_settings import poloniex_API_secret, poloniex_API_key
from non_price_data import google_trends_interest_over_time, initialise_google_session, hash_rate
from CryptocurrencyWebScrapingAndSentimentAnalysis.web_scraper import scrape_subreddits, scrape_forums
from CryptocurrencyWebScrapingAndSentimentAnalysis.sentiment_analysis import analyse_sentiments
SEC_IN_DAY = 86400

SYMBOL_DICTIONARY = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
}

SENTIMENT_KEYWORDS = {
    'BTC': ['bitcoin', 'bitcoins', 'xbt', 'btc', 'Bitcoin', 'Bitcoins', 'BTC', 'XBT'],
    'ETH': ['ethereum', 'ETH'],
}


class Data:   
    def __init__(self, currency_pair, scraper_currency, period, web_flag, start=None, end=None,
                 offset=None, n_days=None, filename=None):
        self.date = []
        self.price = []
        self.close = []
        self.open = []
        self.high = []
        self.low = []
        self.volume = []
        self.time = []
        self.hash_rates = []
        self.google_trend_score = []
        self.web_sentiment_score = []
        self.fractional_close = []
        self.high_low_spread = []
        self.open_close_spread = []
        self.absolute_volatility = []
        self.fractional_volatility = []
        self.fractional_volume = []
        self.exponential_moving_average_1 = []
        self.exponential_moving_average_2 = []
        self.exponential_moving_average_3 = []
        self.exponential_moving_volatility_1 = []
        self.exponential_moving_volatility_2 = []
        self.exponential_moving_volatility_3 = []
        self.exponential_moving_volume_1 = []
        self.exponential_moving_volume_2 = []
        self.exponential_moving_volume_3 = []
        self.kalman_signal = []
        self.scraper_currency = scraper_currency
        self.scraper_score_dates = []
        self.scraper_score_texts = []
        self.classification_score = []
        self.momentum = []
        self.mom_strategy = []
        self.hidden_markov = []

        if web_flag:
            self.candle_input_web(currency_pair, start, end, period)
        else:
            self.candle_input_file(filename, period, offset, n_days)

    def candle_input_file(self, filename, period, offset, n_days):
        candle_array = pd.read_csv(filename).as_matrix()

        end_index = candle_array[-1, 4] - offset * SEC_IN_DAY
        start_index = end_index - n_days * SEC_IN_DAY

        end_index = (np.abs(candle_array[:, 4] - end_index)).argmin()
        start_index = (np.abs(candle_array[:, 4] - start_index)).argmin()

        period_index = period / 300

        self.volume = candle_array[start_index:end_index:period_index, 0]
        self.date = candle_array[start_index:end_index:period_index, 4]
        self.open = candle_array[(start_index + period_index):end_index:period_index, 6]
        self.close = candle_array[(start_index + period_index - 1):end_index:period_index, 5]
        self.close = self.close[-len(self.open):]
        self.high = np.zeros(len(self.close))
        self.low = np.zeros(len(self.close))

        for i in range(int(np.floor(len(self.high) / period_index))):
            loop_start = i * period_index
            self.high[i] = np.max(candle_array[loop_start:loop_start + period_index, 2])
            self.low[i] = np.min(candle_array[loop_start:loop_start + period_index, 3])

    def candle_input_web(self, currency_pair, start, end, period):
        poloniex_session = poloniex(poloniex_API_key, poloniex_API_secret)

        candle_json = poloniex_session.returnChartData(currency_pair, start, end, period)

        candle_length = len(candle_json[u'candleStick'])
        self.volume = nan_array_initialise(candle_length)
        self.date = nan_array_initialise(candle_length)
        self.close = nan_array_initialise(candle_length)
        self.open = nan_array_initialise(candle_length)
        self.high = nan_array_initialise(candle_length)
        self.low = nan_array_initialise(candle_length)

        for loop_counter in range(candle_length):
            self.volume[loop_counter] = float(candle_json[u'candleStick'][loop_counter][u'volume'])
            self.date[loop_counter] = float(candle_json[u'candleStick'][loop_counter][u'date'])
            self.close[loop_counter] = float(candle_json[u'candleStick'][loop_counter][u'close'])
            self.open[loop_counter] = float(candle_json[u'candleStick'][loop_counter][u'open'])
            self.high[loop_counter] = float(candle_json[u'candleStick'][loop_counter][u'high'])
            self.low[loop_counter] = float(candle_json[u'candleStick'][loop_counter][u'low'])

    def extend_candle(self, new_candle):
        for date in new_candle.date:
            if date in self.date:
                trim_candle(new_candle, np.where(new_candle.date == date))

        self.volume = np.concatenate((self.volume, new_candle.date))
        self.date = np.concatenate((self.date, new_candle.date))
        self.open = np.concatenate((self.open, new_candle.open))
        self.close = np.concatenate((self.close, new_candle.close))
        self.high = np.concatenate((self.high, new_candle.high))
        self.low = np.concatenate((self.low, new_candle.low))

    def normalise_data(self):
        self.fractional_close = fractional_change(self.close)

    def calculate_high_low_spread(self):
        self.high_low_spread = self.high - self.low

    def calculate_open_close_spread(self):
        self.open_close_spread = self.close - self.open

    def calculate_fractional_volatility(self):
        self.calculate_high_low_spread()
        self.absolute_volatility = np.abs(self.high_low_spread)
        self.fractional_volatility = fractional_change(self.absolute_volatility)

    def calculate_fractional_volume(self):

        " calculate fractional volume change"

        self.fractional_volume = fractional_change(self.volume)

    def calculate_indicators(self, strategy_dictionary, prior_data_obj=None, non_price_data=False):

        """calculate indicators for training machine learning algorithm"""

        self.exponential_moving_average_1 = exponential_moving_average(
            self.close[:-1],
            strategy_dictionary['windows'])

        self.exponential_moving_average_2 = exponential_moving_average(
            self.close[:-1],
            3 * strategy_dictionary['windows'])

        self.momentum = self.exponential_moving_average_1 - self.exponential_moving_average_2

        self.exponential_moving_volatility_1 = np.sqrt(exponential_moving_average(
            self.momentum ** 2,
            10 * strategy_dictionary['windows']))

        self.momentum_strategy()

        self.hidden_markov = hidden_markov_model(self.close[:-1])

        if non_price_data:
            self.non_price_data(strategy_dictionary, prior_data_obj=prior_data_obj)

    def non_price_data(self, strategy_dictionary, prior_data_obj=None):
        self.hash_rate_data()
        self.google_trend_data(strategy_dictionary)
        self.web_scraping_sentiment_analysis(strategy_dictionary, prior_data_obj=prior_data_obj)

    def hash_rate_data(self):
        hash_rates, dates = hash_rate()
        self.hash_rates = fractional_change(np.interp(self.date, dates, hash_rates))

    def google_trend_data(self, strategy_dictionary):
        pytrend = initialise_google_session()
        search_terms = strategy_dictionary['trading_currencies']

        if 'USDT' in search_terms:
            del search_terms[search_terms == 'USDT']

        self.google_trend_score = np.zeros((len(self.date) - 1, len(search_terms)))

        for i in range(len(search_terms)):
            search_term = SYMBOL_DICTIONARY[search_terms[i]]
            dates, interest = google_trends_interest_over_time(pytrend, [search_term, ])

            self.google_trend_score[:, i] = fractional_change(np.interp(self.date, dates, interest))

    def web_scraping_sentiment_analysis(self, strategy_dictionary, prior_data_obj=None):
        subreddits = ["cryptocurrency", "cryptomarkets", "bitcoin", "bitcoinmarkets", "ethereum"]

        forum_urls = ["https://bitcointalk.org/index.php?board=5.0", "https://bitcointalk.org/index.php?board=7.0",
                      "https://bitcointalk.org/index.php?board=8.0"]
        allowed_domains = ["bitcointalk.org", ]

        if prior_data_obj is None:
            dates, texts = scrape_subreddits(
                subreddits, submission_limit=strategy_dictionary['scraper_page_limit'])
            dates_temp, texts_temp = scrape_forums(
                forum_urls, allowed_domains, max_pages=strategy_dictionary['scraper_page_limit'])

            dates += dates_temp
            texts += texts_temp

            self.scraper_score_dates = dates
            self.scraper_score_texts = texts
        else:
            dates = prior_data_obj.scraper_score_dates
            texts = prior_data_obj.scraper_score_texts

        dates, texts, sentiments = analyse_sentiments(dates, texts, SENTIMENT_KEYWORDS[self.scraper_currency])

        sentiments, dates = sort_arrays_by_first(dates, sentiments)

        self.web_sentiment_score = fractional_change(np.interp(self.date, dates, sentiments))

        logging.getLogger().setLevel(logging.WARNING)

    def momentum_strategy(self):

        """simple momentum strategy for comparison"""

        self.mom_strategy = self.momentum / self.exponential_moving_volatility_1


class TradingTargets:

    """Class to produce targets for machine learning algorithms to output"""

    def __init__(self, normalise_data_obj):
        self.fractional_close = normalise_data_obj.fractional_close
        self.high = normalise_data_obj.high
        self.strategy_score = np.full([len(self.fractional_close)], np.nan)
        self.buy_sell = []

    def ideal_strategy_score(self, strategy_dictionary):

        """score based on max or min value before next trend change"""

        fractional_close_length = len(self.fractional_close)

        self.strategy_score = np.ones(fractional_close_length)

        for index in range(fractional_close_length):
            while_counter = 0
            net_change = 1.0
            last_change = 1
            while index + while_counter < fractional_close_length:
                net_change *= self.fractional_close[index + while_counter]

                if net_change < 1:

                    if net_change < last_change:
                        score = net_change

                    if last_change > 1:
                        self.strategy_score[while_counter] = score
                        break

                elif net_change > 1:

                    if net_change > last_change:
                        score = net_change

                    if last_change < 1:
                        self.strategy_score[while_counter] = score
                        break

                last_change = net_change
                while_counter += 1

        self.normalise_with_std(strategy_dictionary['windows'])

        self.strategy_score[np.isnan(self.strategy_score)] = 0

    def n_steps_ahead_score(self, n_steps):

        """training target based on average score n steps ahead"""

        fractional_close_length = len(self.fractional_close)

        self.strategy_score = np.zeros(fractional_close_length)

        for index in range(fractional_close_length):

            end_index = np.minimum(index + n_steps + 1, fractional_close_length - 1)

            self.strategy_score[index] = np.product(self.fractional_close[index+1:end_index])

        self.strategy_score[np.isnan(self.strategy_score)] = 1

        self.normalise_with_std(n_steps)

    def normalise_with_std(self, window):

        self.strategy_score\
            = self.strategy_score / np.sqrt(exponential_moving_average(self.strategy_score ** 2, window))

    def convert_score_to_classification_target(self):

        """convert continuous target to discrete classfication target"""

        self.classification_score = np.zeros(len(self.strategy_score))
        self.classification_score[self.strategy_score > 1] = 1
        self.classification_score[self.strategy_score < 1] = -1
        self.classification_score = self.classification_score.astype(int)


def effective_fee(strategy_dictionary):

    """effective fee as a ratio of a transaction"""

    return 1 - strategy_dictionary['transaction_fee'] - strategy_dictionary['bid_ask_spread']


def trim_candle(candle, index):

    """from candle from candlestick data"""

    np.delete(candle.date, index)
    np.delete(candle.open, index)
    np.delete(candle.close, index)
    np.delete(candle.high, index)
    np.delete(candle.low, index)


def fractional_change(price):
    replacement = min(price[price != 0]) / 1e3
    temp = price
    temp[temp == 0] = replacement
    return price[1:] / price[:-1]


def exponential_moving_average(data, window):
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        ema = np.convolve(data, weights, mode='full')[:len(data)]
        ema[:window] = ema[window]
        return ema


def staggered_input(input_vector, offset):
    fractional_price_array = input_vector[offset:]
    for index in range(1, offset):
        fractional_price_array = np.vstack((fractional_price_array, input_vector[offset - index:-index]))

    return fractional_price_array


def calculate_data_length(start, end, period):
    return int((end - start) / period)


def nan_array_initialise(size):
    array = np.empty((size,))
    array[:] = np.NaN
    return array


def generate_training_variables(data_obj, strategy_dictionary, prior_data_obj=None):

    """return all of the training variables"""

    trading_targets = TradingTargets(data_obj)

    if strategy_dictionary['target_score'] == 'ideal_strategy':
        trading_targets.ideal_strategy_score(strategy_dictionary)

    elif strategy_dictionary['target_score'] == 'n_steps':
        trading_targets.n_steps_ahead_score(strategy_dictionary['target_step'])

    trading_targets.convert_score_to_classification_target()

    data_obj.calculate_indicators(strategy_dictionary, prior_data_obj=prior_data_obj)

    fitting_inputs = np.vstack((
        data_obj.momentum,
        data_obj.exponential_moving_volatility_1,
        data_obj.hidden_markov,
        #data_obj.kalman_signal,
        # data_obj.close[:-1],
        #data_obj.open[:-2],
        # data_obj.high[:-1],
        # data_obj.low[:-1],
        # pad_nan(data_obj.close[:-2], 1),
        # pad_nan(data_obj.open[:-2], 1),
        # pad_nan(data_obj.high[:-2], 1),
        # pad_nan(data_obj.low[:-2], 1),
        # pad_nan(data_obj.close[:-3], 2),
        # pad_nan(data_obj.open[:-3], 2),
        # pad_nan(data_obj.high[:-3], 2),
        # pad_nan(data_obj.low[:-3], 2),
        #data_obj.google_trend_score[:-1].T,
        #data_obj.web_sentiment_score[:-1],
        #data_obj.hash_rates[:-1]
        ))

    fitting_inputs = fitting_inputs.T

    fitting_inputs_scaled = scale(fitting_inputs)

    continuous_targets = scale(trading_targets.strategy_score[1:])
    classification_targets = scale(trading_targets.classification_score[1:])

    return fitting_inputs_scaled, continuous_targets, classification_targets


def preprocessing_inputs(strategy_dictionary, fitting_inputs_scaled):
    if strategy_dictionary['preprocessing'] == 'PCA':
        fitting_inputs_scaled = pca_transform(fitting_inputs_scaled)

    if strategy_dictionary['preprocessing'] == 'FastICA':
        fitting_inputs_scaled, strategy_dictionary = fast_ica_transform(strategy_dictionary, fitting_inputs_scaled)

    return fitting_inputs_scaled, strategy_dictionary


def pad_nan(vector, n):
    pad_vector = np.zeros(n)
    return np.hstack((pad_vector, vector))


def imputer_transform(data):
    imputer = Imputer()
    imputer.fit(data)
    return imputer.transform(data)


def pca_transform(fitting_inputs_scaled):
    pca = PCA()
    pca.fit(fitting_inputs_scaled)

    return pca.transform(fitting_inputs_scaled)


def fast_ica_transform(strategy_dictionary, fitting_inputs_scaled):

    try:
        ica = FastICA()
        ica.fit(fitting_inputs_scaled)

        fitting_inputs_scaled = ica.transform(fitting_inputs_scaled)

    except:
        strategy_dictionary['preprocessing'] = 'None'

    return fitting_inputs_scaled, strategy_dictionary


def train_test_indices(input_data, train_factor):
    data_length = len(input_data)
    train_indices_local = range(0, int(data_length * train_factor))
    test_indices_local = range(train_indices_local[-1] + 1, data_length)

    return train_indices_local, test_indices_local


def train_test_validation_indices(input_data, ratios):
    train_factor = ratios[0]
    val_factor = ratios[1]
    data_length = len(input_data)
    train_indices_local = range(0, int(data_length * train_factor))
    validation_indices_local = range(train_indices_local[-1] + 1, int(data_length * (train_factor + val_factor)))

    test_indices_local = range(validation_indices_local[-1] + 1, data_length)

    return train_indices_local, test_indices_local, validation_indices_local


def kalman_filter(input_price):

    """kalman filter to be used as training input if required"""

    n_iter = len(input_price)
    vector_size = (n_iter,)

    q = 1E-5

    post_estimate = np.zeros(vector_size)
    p = np.zeros(vector_size)
    post_estimate_minus = np.zeros(vector_size)
    pminus = np.zeros(vector_size)
    k_var = np.zeros(vector_size)

    r = 0.1 ** 2

    post_estimate[0] = input_price[0]
    p[0] = 1.0

    for k in range(1, n_iter):
        post_estimate_minus[k] = post_estimate[k - 1]
        pminus[k] = p[k - 1] + q

        k_var[k] = pminus[k] / (pminus[k] + r)
        post_estimate[k] = post_estimate_minus[k] + k_var[k] * (input_price[k] - post_estimate_minus[k])
        p[k] = (1 - k_var[k]) * pminus[k]

    return post_estimate


def sort_arrays_by_first(y_input, x_input):
    return [x for (y, x) in sorted(zip(y_input, x_input))], [y for (y, x) in sorted(zip(y_input, x_input))]


def hidden_markov_model(X):

    """fit hidden markov model """

    X = X.reshape(-1, 1)

    remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    remodel.fit(X)
    return remodel.predict(X)



