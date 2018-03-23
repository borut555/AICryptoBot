import numpy as np
from time import time
from data_input_processing import Data, train_test_validation_indices, generate_training_variables
from strategy_evaluation import post_process_training_results, output_strategy_results
from machine_learning import random_forest_fitting, svm_fitting, adaboost_fitting, gradient_boosting_fitting,\
    extra_trees_fitting, tensorflow_fitting, tensorflow_sequence_fitting
from sklearn.preprocessing import StandardScaler

SEC_IN_DAY = 86400


def meta_fitting(fitting_inputs, fitting_targets, strategy_dictionary):
    error = []
    model = []
    train_indices, test_indices, validation_indices = train_test_validation_indices(
        fitting_inputs,
        strategy_dictionary['train_test_validation_ratios'])

    target_scaler = StandardScaler()

    if strategy_dictionary['regression_mode'] == 'regression':
        fitting_targets = target_scaler.fit_transform(fitting_targets.reshape(-1, 1)).ravel()

    if strategy_dictionary['ml_mode'] == 'svm':
        model, error = svm_fitting(
            fitting_inputs,
            fitting_targets,
            train_indices,
            strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'randomforest':
        model, error = random_forest_fitting(
            fitting_inputs,
            fitting_targets,
            train_indices,
            strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'adaboost':
        model, error = adaboost_fitting(
            fitting_inputs,
            fitting_targets,
            train_indices,
            strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'gradientboosting':
        model, error = gradient_boosting_fitting(
            fitting_inputs,
            fitting_targets,
            train_indices,
            strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'extratreesfitting':
        model, error = extra_trees_fitting(
            fitting_inputs,
            fitting_targets,
            train_indices,
            strategy_dictionary)

    if len(test_indices) != 0:
        training_strategy_score = model.predict(fitting_inputs[train_indices, :])
        fitted_strategy_score = model.predict(fitting_inputs[test_indices, :])
        validation_strategy_score = model.predict(fitting_inputs[validation_indices, :])

    else:
        fitted_strategy_score = []
        validation_strategy_score = []

    if strategy_dictionary['regression_mode'] == 'regression':
        fitting_dictionary = {
            'training_strategy_score': training_strategy_score,
            'fitted_strategy_score': fitted_strategy_score,
            'validation_strategy_score':validation_strategy_score,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'validation_indices': validation_indices,
            'error': error}

    elif strategy_dictionary['regression_mode'] == 'classification':
        fitting_dictionary = {
            'training_strategy_score': training_strategy_score,
            'fitted_strategy_score': fitted_strategy_score,
            'validation_strategy_score': validation_strategy_score,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'validation_indices': validation_indices,
            'error': error}

    return fitting_dictionary


def input_processing(data_to_predict_local, strategy_dictionary):

    """retrieve and process continuous and classification targets"""

    fitting_inputs, continuous_targets, classification_targets = generate_training_variables(
        data_to_predict_local,
        strategy_dictionary)

    return fitting_inputs, continuous_targets, classification_targets


def tic():
    t = time()
    return lambda: (time() - t)


def retrieve_data(ticker, scraper_currency, strategy_dictionary, filename):
    data_local = None
    #while data_local is None:
    #    try:
    if strategy_dictionary['web_flag']:
        end = time() - strategy_dictionary['offset'] * SEC_IN_DAY

        start = end - SEC_IN_DAY * strategy_dictionary['n_days']

        data_local = Data(
            ticker,
            scraper_currency,
            strategy_dictionary['candle_size'],
            strategy_dictionary['web_flag'],
            start=start,
            end=end,
        )

    else:
        data_local = Data(
            ticker, scraper_currency,
            strategy_dictionary['candle_size'],
            strategy_dictionary['web_flag'],
            offset=strategy_dictionary['offset'],
            filename=filename,
            n_days=strategy_dictionary['n_days'])

    #   except Exception:
    #       pass

    data_local.normalise_data()

    return data_local


def offset_scan_validation(strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets, offsets):

    """repeat fitting at a range of earlier offset start times to check for overfitting"""

    strategy_dictionary['plot_flag'] = False
    strategy_dictionary['ouput_flag'] = True

    total_error = 0
    total_profit = 0

    for offset in offsets:
        strategy_dictionary['offset'] = offset
        fitting_dictionary,  profit_factor, test_profit_factor, _ = fit_strategy(
            strategy_dictionary,
            data_to_predict,
            fitting_inputs,
            fitting_targets)
        total_error += fitting_dictionary['error'] / len(offsets)
        total_profit += profit_factor

    underlined_output('Averages: ')
    print ('Total profit: ', total_profit)
    print ('Average error: ', total_error)


def tensorflow_offset_scan_validation(strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets, offsets):

    """repeat tensorflow fitting at a range of earlier offset start times to check for overfitting"""

    strategy_dictionary['plot_flag'] = False
    strategy_dictionary['ouput_flag'] = True
    
    total_error = 0
    total_profit = 0

    for offset in offsets:
        strategy_dictionary['offset'] = offset
        fitting_dictionary, error, profit_fraction = fit_tensorflow(
            strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets)
        total_error += error
        total_profit += profit_fraction

    underlined_output('Averages: ')
    print ('Total profit: ', total_profit)
    print ('Average error: ', total_error)


def import_data(strategy_dictionary):

    data_to_predict = retrieve_data(
        strategy_dictionary['ticker_1'],
        strategy_dictionary['scraper_currency_1'],
        strategy_dictionary,
        strategy_dictionary['filename1'])

    return data_to_predict


def fit_strategy(strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets):

    """fit machine learning algorithm to data and return predictions and profit"""

    toc = tic()

    fitting_dictionary = meta_fitting(fitting_inputs, fitting_targets, strategy_dictionary)

    fitting_dictionary, strategy_dictionary = post_process_training_results(
        strategy_dictionary,
        fitting_dictionary,
        data_to_predict)

    profit_factor, test_profit_factor = output_strategy_results(
        strategy_dictionary,
        fitting_dictionary,
        data_to_predict,
        toc)

    return fitting_dictionary, profit_factor, test_profit_factor, strategy_dictionary


def fit_tensorflow(strategy_dictionary, data_to_predict, fitting_inputs, fitting_targets):

    """fit with tensorflow and return predictions and profit"""

    toc = tic()

    train_indices, test_indices, validation_indices = train_test_validation_indices(
        fitting_inputs, strategy_dictionary['train_test_validation_ratios'])

    if strategy_dictionary['sequence_flag']:
        fitting_dictionary, error = tensorflow_sequence_fitting(
            train_indices,
            test_indices,
            validation_indices,
            fitting_inputs,
            fitting_targets)

    else:
        fitting_dictionary, error = tensorflow_fitting(
            train_indices,
            test_indices,
            validation_indices,
            fitting_inputs,
            fitting_targets)

    fitting_dictionary['train_indices'] = train_indices
    fitting_dictionary['test_indices'] = test_indices
    fitting_dictionary['validation_indices'] = validation_indices

    fitting_dictionary, strategy_dictionary = post_process_training_results(
        strategy_dictionary,
        fitting_dictionary,
        data_to_predict)

    profit_factor = output_strategy_results(strategy_dictionary, fitting_dictionary, data_to_predict, toc)
    return fitting_dictionary, error, profit_factor


def underlined_output(string):

    """underline printed output"""

    print (string)
    print ('----------------------')
    print ('\n')


def normalise_and_centre_score(strategy_score, up_threshold, low_threshold):

    """normalise and centre score when fitting thresholds"""

    temp_score = strategy_score
    temp_score[temp_score > up_threshold] = up_threshold
    temp_score[temp_score < -up_threshold] = -up_threshold
    temp_score[abs(temp_score) < low_threshold] = 0
    temp_score = temp_score / (2 * up_threshold)
    temp_score = temp_score + 0.5

    return temp_score



