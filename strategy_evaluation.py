import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale


def strategy_profit(
        strategy_score,
        fractional_price,
        strategy_dictionary,
        low_threshold,
        up_threshold):

    """calculate net profit of trading strategy """

    buy_sell_length = len(strategy_score)
    portfolio_value = np.ones(buy_sell_length)
    cash_value = np.zeros(buy_sell_length)
    crypto_value = np.zeros(buy_sell_length)
    effective_fee_factor = (strategy_dictionary['transaction_fee'] + 0.5 * strategy_dictionary['bid_ask_spread'])

    n_trades = 0
    curr_val = 1
    stop_val = 1

    strategy_score = normalise_and_centre_score(
        strategy_score,
        up_threshold,
        low_threshold)
        #max_scale=strategy_dictionary['max_scale'])

    cash_value[0] = 1 - strategy_score[0]
    crypto_value[0] = strategy_score[0] * (1 - effective_fee_factor)

    for index in range(1, buy_sell_length):
        crypto_value[index] = crypto_value[index - 1] * fractional_price[index - 1]
        cash_value[index] = cash_value[index - 1]
        portfolio_value[index] = crypto_value[index] + cash_value[index]

        curr_val *= fractional_price[index - 1]

        score_step = (strategy_score[index] - strategy_score[index - 1])

        score_step = portfolio_value[index] * score_step

        #Stoploss
        stop_val = max(stop_val, curr_val)

        if 0 < (stop_val - curr_val) / stop_val < (1 - strategy_dictionary['stop_loss']):
            score_step = - 10

            if np.prod(fractional_price[index-1]) > 1:
                stop_val = curr_val

        if score_step > 0:

            if score_step > cash_value[index]:
                score_step = cash_value[index]

            effective_fee = effective_fee_factor * score_step

            cash_value[index] = cash_value[index] - score_step
            crypto_value[index] = crypto_value[index] + score_step - effective_fee

            n_trades += 1

        elif score_step < 0:

            score_step = abs(score_step)

            if score_step > crypto_value[index]:
                score_step = crypto_value[index]

            effective_fee = effective_fee_factor * score_step

            cash_value[index] = cash_value[index] + score_step - effective_fee
            crypto_value[index] = crypto_value[index] - score_step

            n_trades += 1

    return portfolio_value, n_trades, cash_value, crypto_value, strategy_score


def normalise_and_centre_score(strategy_score, up_threshold, low_threshold, max_scale=False):

    """normalise and centre score when fitting thresholds"""

    temp_score = minmax_scale(strategy_score)

    temp_score = temp_score - 0.5
    temp_score[temp_score > up_threshold] = up_threshold
    temp_score[temp_score < -up_threshold] = -up_threshold
    temp_score[abs(temp_score) < low_threshold] = 0
    temp_score += 0.5

    if max_scale:
        temp_score = minmax_scale(temp_score)

    return temp_score


def fit_trade_threshold(strategy_score, fractional_price, strategy_dictionary):

    """ fit minimum signal change to execute trade """

    threshold_range = np.linspace(0, 1, 30)

    best_profit = -1
    best_up_threshold = 0
    best_low_threshold = 0

    for up_threshold in threshold_range:

        low_threshold_range = threshold_range[threshold_range < up_threshold]

        for low_threshold in low_threshold_range:
            profit_vector, n_trades, _, _, _ = strategy_profit(
                strategy_score,
                fractional_price,
                strategy_dictionary,
                low_threshold,
                up_threshold)
            profit = strategy_profit_score(profit_vector, n_trades)

            if profit > best_profit and n_trades != 0:
                best_low_threshold = low_threshold
                best_up_threshold = up_threshold
                best_profit = profit

        strategy_dictionary['low_threshold'] = best_low_threshold
        strategy_dictionary['up_threshold'] = best_up_threshold

    return strategy_dictionary


def post_process_training_results(strategy_dictionary, fitting_dictionary, data):

    """return fitting dictionary containing training parameters"""

    strategy_dictionary = fit_trade_threshold(
        fitting_dictionary['validation_strategy_score'],
        data.fractional_close[fitting_dictionary['validation_indices']],
        strategy_dictionary)

    """
    fitting_dictionary['portfolio_value'], fitting_dictionary['n_trades'], cash_value, crypto_value, fitting_dictionary['test_strategy_score']\
        = strategy_profit(
        fitting_dictionary['test_strategy_score'],
        data.fractional_close[fitting_dictionary['test_indices']],
        strategy_dictionary,
        strategy_dictionary['low_threshold'],
        strategy_dictionary['up_threshold'])
    """
    fitting_dictionary['portfolio_value'], fitting_dictionary['n_trades'], cash_value, crypto_value, fitting_dictionary['fitted_strategy_score']\
        = strategy_profit(
        fitting_dictionary['fitted_strategy_score'],
        data.fractional_close[fitting_dictionary['test_indices']],
        strategy_dictionary,
        strategy_dictionary['low_threshold'],
        strategy_dictionary['up_threshold'])
    
    
    fitting_dictionary['validation_portfolio_value'],\
        fitting_dictionary['validation_trades'],\
        _,\
        _,\
        fitting_dictionary['validation_strategy_score']\
        = strategy_profit(
        fitting_dictionary['validation_strategy_score'],
        data.fractional_close[fitting_dictionary['validation_indices']],
        strategy_dictionary,
        strategy_dictionary['low_threshold'],
        strategy_dictionary['up_threshold'])

    return fitting_dictionary, strategy_dictionary


def strategy_profit_score(strategy_profit_local, number_of_trades):

    """evaluate value added by the trading strategy overall"""

    profit_fraction = strategy_profit_local[-1] / np.min(strategy_profit_local)
    if number_of_trades == 0:
        profit_fraction = -profit_fraction
    return profit_fraction


def draw_down(strategy_profit_local):

    """find maximum drawdown of strategy"""

    draw_down_temp = np.diff(strategy_profit_local)
    draw_down_temp[draw_down_temp > 0] = 0
    return np.mean(draw_down_temp)


def profit_factor(portfolio_value, price):

    """calculate profit of strategy compared to buy and hold """

    return portfolio_value[-1] * price[0] / (portfolio_value[0] * price[-1]) - 1


def output_strategy_results(strategy_dictionary, fitting_dictionary, data_to_predict, toc, momentum_dict=None):

    """print or plot results of machine learning fitting"""

    val_data = data_to_predict.close[fitting_dictionary['validation_indices']]
    prediction_data = data_to_predict.close[fitting_dictionary['test_indices']]

    profit = profit_factor(
            fitting_dictionary['portfolio_value'],
            prediction_data)

    val_profit = profit_factor(
            fitting_dictionary['validation_portfolio_value'],
            val_data)

    if strategy_dictionary['output_flag']:
        print ("Fitting time: ", toc())
        
        print ("Raw profit", fitting_dictionary['portfolio_value'][-1])
        print ("Fractional profit compared to buy and hold: ", profit)
        print ("Mean squared error: ", fitting_dictionary['error'])
        print ("Window: ", strategy_dictionary['windows'])
        print ("Target step :", strategy_dictionary['target_step'])
        print ("Fitting model: ", strategy_dictionary['ml_mode'])
        print ("Regression/classification: ", strategy_dictionary['regression_mode'])
        print ("Number of trades: ", fitting_dictionary['n_trades'])
        print ("Offset: ", strategy_dictionary['offset'])

        if momentum_dict is not None:
            print ("Simple Momentum Profit: ", profit_factor, momentum_dict['portfolio_value'],prediction_data)

        print ("\n")

    if strategy_dictionary['plot_flag']:
        plt.figure(1)
        close_price, = plt.plot(prediction_data)
        portfolio_value, = plt.plot(
            prediction_data[0] * fitting_dictionary['portfolio_value'])

        mom_strategy = []
        if momentum_dict is not None:
            mom_strategy, = plt.plot(prediction_data[0] * momentum_dict['portfolio_value'])

        plt.legend([close_price, portfolio_value, mom_strategy], ['Close Price', 'Portfolio Value', 'Momentum Strategy'])
        plt.xlabel('Candle number')
        plt.ylabel('Exchange rate')

        plt.figure(2)
        validation_score, = plt.plot(np.squeeze(fitting_dictionary['validation_strategy_score']), )

        momentum_score = []
        if momentum_dict is not None:
            momentum_score, = plt.plot(np.squeeze(momentum_dict['validation_score']), )

        plt.legend([validation_score, momentum_score], ['Validation Score', 'Momentum Score'])

        plt.show()
    
    return profit, val_profit


def simple_momentum_comparison(data_obj, strategy_dictionary, fitting_dictionary):

    """implement simple momentum strategy for comparison to machine learning"""

    validation_indices = fitting_dictionary['validation_indices']
    try:
        validation_indices.remove(len(data_obj.fractional_close))

    except:
        pass

    validation_indices\
        = [v for v in validation_indices if v < min(len(data_obj.fractional_close), len(data_obj.mom_strategy))]

    test_data = data_obj.fractional_close[fitting_dictionary['test_indices']]
    val_data = data_obj.fractional_close[validation_indices]

    test_momentum = data_obj.mom_strategy[fitting_dictionary['test_indices']]
    val_momentum = data_obj.mom_strategy[validation_indices]

    momentum_dictionary = {}

    strategy_dictionary = fit_trade_threshold(
        val_momentum,
        val_data,
        strategy_dictionary)

    momentum_dictionary['portfolio_value'],\
        momentum_dictionary['n_trades'],\
        _,\
        _,\
        momentum_dictionary['validation_score']\
        = strategy_profit(
        test_momentum,
        test_data,
        strategy_dictionary,
        strategy_dictionary['low_threshold'],
        strategy_dictionary['up_threshold'])

    return momentum_dictionary
