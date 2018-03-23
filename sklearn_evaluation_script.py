import numpy as np
from random import choice, randint
from trading_strategy_fitting import fit_strategy, offset_scan_validation, tic, underlined_output, import_data,\
    input_processing
from data_input_processing import preprocessing_inputs
from strategy_evaluation import output_strategy_results, simple_momentum_comparison


def random_search(strategy_dictionary_local, n_iterations, data_local, toc):

    """random search to find optimum machine learning algorithm and preprocessing"""

    fitting_inputs_local, continuous_targets, classification_targets = input_processing(
        data_local,
        strategy_dictionary_local)

    counter = 0
    error = 1e5
    fitting_targets_local = []
    fitting_dictionary_optimum = []
    strategy_dictionary_optimum = []
    while counter < n_iterations:
        counter += 1
        strategy_dictionary_local = randomise_dictionary_inputs(strategy_dictionary_local)

        if strategy_dictionary_local['regression_mode'] == 'classification':
            fitting_targets_local = classification_targets.astype(int)
        elif strategy_dictionary_local['regression_mode'] == 'regression':
            fitting_targets_local = continuous_targets

        fitting_inputs_local, strategy_dictionary_local = preprocessing_inputs(
            strategy_dictionary_local,
            fitting_inputs_local)

        fitting_dictionary_local, _, _, strategy_dictionary_local = fit_strategy(
            strategy_dictionary_local,
            data_local,
            fitting_inputs_local,
            fitting_targets_local)

        error_loop = fitting_dictionary_local['error']

        if error_loop < error and fitting_dictionary_local['n_trades'] != 0:
            error = error_loop
            strategy_dictionary_optimum = strategy_dictionary_local
            fitting_dictionary_optimum = fitting_dictionary_local

    if strategy_dictionary_optimum:
        profit, val_profit = output_strategy_results(
            strategy_dictionary_optimum,
            fitting_dictionary_optimum,
            data_local,
            toc)

    else:
        val_profit = -2

    return strategy_dictionary_optimum,\
        fitting_dictionary_optimum,\
        fitting_inputs_local,\
        fitting_targets_local,\
        data_local,\
        val_profit


def randomise_dictionary_inputs(strategy_dictionary_local):

    """ generate parameters for next step of random search """

    strategy_dictionary_local['ml_mode'] = choice([
        'svm',
        'randomforest',
        'adaboost',
        'gradientboosting',
        'extratreesfitting'
    ])

    strategy_dictionary_local['preprocessing'] = choice(['PCA', 'FastICA', 'None'])
    return strategy_dictionary_local


def randomise_time_inputs(strategy_dictionary_local):

    """ generate time parameters for next step of random search """

    window = strategy_dictionary['n_days'] * 100 * 0.9

    strategy_dictionary_local['windows'] = randint(1, window / 10)

    strategy_dictionary_local['target_step'] = randint(1, window)

    return strategy_dictionary_local


def fit_time_scale(strategy_dictionary_input, search_iterations_local, time_iterations, date, toc):

    """ fit timescale variables"""

    counter = 0
    strategy_dictionary_optimum = []
    optimum_profit = -2

    while counter < time_iterations:

        strategy_dictionary_input = randomise_time_inputs(strategy_dictionary_input)

        strategy_dictionary_local,\
            fitting_dictionary_local,\
            fitting_inputs_local,\
            fitting_targets_local,\
            data_local,\
            test_profit\
            = random_search(
                strategy_dictionary_input,
                search_iterations_local,
                date,
                toc)

        if test_profit > optimum_profit:
            optimum_profit = test_profit
            strategy_dictionary_optimum = strategy_dictionary_local
            fitting_dictionary_optimum = fitting_dictionary_local

        counter += 1

    return strategy_dictionary_optimum,\
        fitting_dictionary_optimum,\
        fitting_inputs_local,\
        fitting_targets_local,\
        data_local


if __name__ == '__main__':
    
    print ("start")
    toc = tic()
       

    strategy_dictionary = {
        'trading_currencies': ['BTC', 'ETH'],
        'ticker_1': 'BTC_ETH',
        'scraper_currency_1': 'ETH',
        'candle_size': 300,
        'n_days': 10,
        'offset': 0,
        'bid_ask_spread': 0.003,
        'transaction_fee': 0.0015,
        'train_test_validation_ratios': [0.5, 0.2, 0.3],
        'output_flag': True,
        'plot_flag': False,
        'plot_last': True,
        'ml_iterations': 10,
        'target_score': 'n_steps',
        'web_flag': False,
        'filename1': "BTC_ETH.csv",
        'regression_mode': 'regression',
        'momentum_compare': True,
        'fit_time': False,
        'target_step': 200,
        'windows': 20,
        'stop_loss': 0.07
    }
    
    print (strategy_dictionary)

    data = import_data(strategy_dictionary)    

    search_iterations = 5
    time_iterations = 20

    if strategy_dictionary['fit_time']:
        strategy_dictionary, fitting_dictionary, fitting_inputs, fitting_targets, data_to_predict = fit_time_scale(
            strategy_dictionary,
            search_iterations,
            time_iterations,
            data,
            toc)

    else:
        strategy_dictionary, fitting_dictionary, fitting_inputs, fitting_targets, data_to_predict, test_profit \
            = random_search(
            strategy_dictionary,
            search_iterations,
            data,
            toc)

    underlined_output('Best strategy fit')

    if strategy_dictionary['plot_last']:
        strategy_dictionary['plot_flag'] = True

    output_strategy_results(
        strategy_dictionary,
        fitting_dictionary,
        data_to_predict,
        toc,
        momentum_dict=simple_momentum_comparison(data_to_predict, strategy_dictionary, fitting_dictionary))

    underlined_output('Offset validation')
    offsets = np.linspace(0, 300, 5)

    offset_scan_validation(strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets, offsets)
