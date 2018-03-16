from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, LSTM, Input
from keras.models import Model, Sequential
from bayes_opt import BayesianOptimization
import numpy as np


def svm_fitting(input_data, target_data, train_indices, strategy_dictionary):

    """ find optimum hyperparameters for svm"""

    param_set = {'C': (0.001, 100), 'gamma': (0.0001, 0.1)}

    if strategy_dictionary['regression_mode'] == 'regression':
        bo = BayesianOptimization(
            lambda C, gamma:
            bayesian_wrapper(
                SVR(C=C, gamma=gamma, kernel='rbf'),
                train_indices,
                input_data,
                target_data),
            param_set)

    elif strategy_dictionary['regression_mode'] == 'classification':

        bo = BayesianOptimization(
            lambda C, gamma:
            bayesian_wrapper(
                SVC(C=C, gamma=gamma, kernel='rbf'),
                train_indices,
                input_data,
                target_data),
            param_set)

    bo.maximize(n_iter=strategy_dictionary['ml_iterations'])

    result_dict = bo.res['max']

    error = 1 / result_dict['max_val']

    if strategy_dictionary['regression_mode'] == 'regression':
        model = SVR(C=result_dict['max_params']['C'], gamma=result_dict['max_params']['gamma'], kernel='rbf')

    elif strategy_dictionary['regression_mode'] == 'classification':
        model = SVC(C=result_dict['max_params']['C'], gamma=result_dict['max_params']['gamma'], kernel='rbf')

    model.fit(input_data[train_indices, :], target_data[train_indices])

    return model, error


def bayesian_wrapper(clf, train_indices, input_data, target_data):

    """ wrapper for bayesian optimization """

    clf.fit(input_data[train_indices, :], target_data[train_indices])

    prediction = clf.predict(input_data[train_indices, :])

    val = mean_squared_error(
        target_data[train_indices],
        prediction,
    ).mean()

    return 1 / val


def random_forest_fitting(
        input_data,
        target_data,
        train_indices,
        strategy_dictionary):

    """ find optimum hyperparameters for random forest"""

    param_set = {'n_estimators': (2, 1000),
                 'max_depth': (1, 10),
                 'max_features': (1, 2)}

    if strategy_dictionary['regression_mode'] == 'regression':
        bo = BayesianOptimization(
            lambda n_estimators, max_depth, max_features:
            bayesian_wrapper(
                RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    max_features=int(max_features)),
                train_indices,
                input_data,
                target_data),
            param_set)

    elif strategy_dictionary['regression_mode'] == 'classification':
        bo = BayesianOptimization(
            lambda n_estimators, max_depth, max_features:
            bayesian_wrapper(
                RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    max_features=int(max_features)),
                train_indices,
                input_data,
                target_data),
            param_set)

    bo.maximize(n_iter=strategy_dictionary['ml_iterations'])

    result_dict = bo.res['max']

    error = 1 / result_dict['max_val']

    if strategy_dictionary['regression_mode'] == 'regression':
        model = RandomForestRegressor(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            max_depth=int(result_dict['max_params']['max_depth']),
            max_features=int(result_dict['max_params']['max_features']))

    elif strategy_dictionary['regression_mode'] == 'classification':
        model = RandomForestClassifier(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            max_depth=int(result_dict['max_params']['max_depth']),
            max_features=int(result_dict['max_params']['max_features']))

    model.fit(input_data[train_indices, :], target_data[train_indices])

    return model, error


def adaboost_fitting(
        input_data,
        target_data,
        train_indices,
        strategy_dictionary):

    """ find optimum hyperparameters for adaboost"""

    param_set = {'learning_rate': (0.1, 1.0),
                 "n_estimators": (2, 1000)}

    if strategy_dictionary['regression_mode'] == 'regression':
        bo = BayesianOptimization(
            lambda n_estimators, learning_rate:
            bayesian_wrapper(
                AdaBoostRegressor(
                    n_estimators=int(n_estimators),
                    learning_rate=learning_rate),
                train_indices,
                input_data,
                target_data),
            param_set)

    elif strategy_dictionary['regression_mode'] == 'classification':
        bo = BayesianOptimization(
            lambda n_estimators, learning_rate:
            bayesian_wrapper(
                AdaBoostClassifier(
                    n_estimators=int(n_estimators),
                    learning_rate=learning_rate),
                train_indices,
                input_data,
                target_data),
            param_set)

    bo.maximize(n_iter=strategy_dictionary['ml_iterations'])

    result_dict = bo.res['max']

    error = 1 / result_dict['max_val']

    if strategy_dictionary['regression_mode'] == 'regression':
        model = AdaBoostRegressor(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            learning_rate=result_dict['max_params']['learning_rate'])

    elif strategy_dictionary['regression_mode'] == 'classification':
        model = AdaBoostClassifier(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            learning_rate=result_dict['max_params']['learning_rate'])

    model.fit(input_data[train_indices, :], target_data[train_indices])

    return model, error


def gradient_boosting_fitting(
        input_data,
        target_data,
        train_indices,
        strategy_dictionary):

    """ find optimum hyperparameters for gradient boosting"""

    param_set = {'n_estimators': (2, 1000),
                 'max_depth': (1, 3),
                 'learning_rate': (0.1, 1.0)}

    if strategy_dictionary['regression_mode'] == 'regression':
        bo = BayesianOptimization(
            lambda n_estimators, learning_rate, max_depth:
            bayesian_wrapper(
                GradientBoostingRegressor(
                    n_estimators=int(n_estimators),
                    learning_rate=learning_rate,
                    max_depth=int(max_depth)),
                train_indices,
                input_data,
                target_data),
            param_set)

    elif strategy_dictionary['regression_mode'] == 'classification':
        bo = BayesianOptimization(
            lambda n_estimators, learning_rate, max_depth:
            bayesian_wrapper(
                GradientBoostingClassifier(
                    n_estimators=int(n_estimators),
                    learning_rate=learning_rate,
                    max_depth = int(max_depth)),
                train_indices,
                input_data,
                target_data),
            param_set)

    bo.maximize(n_iter=strategy_dictionary['ml_iterations'])

    result_dict = bo.res['max']

    error = 1 / result_dict['max_val']

    if strategy_dictionary['regression_mode'] == 'regression':
        model = GradientBoostingRegressor(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            learning_rate=result_dict['max_params']['learning_rate'],
            max_depth=int(result_dict['max_params']['max_depth']))

    elif strategy_dictionary['regression_mode'] == 'classification':
        model = GradientBoostingClassifier(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            learning_rate=result_dict['max_params']['learning_rate'],
            max_depth=int(result_dict['max_params']['max_depth']))

    model.fit(input_data[train_indices, :], target_data[train_indices])

    return model, error


def extra_trees_fitting(
        input_data,
        target_data,
        train_indices,
        strategy_dictionary):

    """ find optimum hyperparameters for extra trees"""

    param_set = {'n_estimators': (2, 1000),
                 'max_depth': (1, 10)}

    if strategy_dictionary['regression_mode'] == 'regression':
        bo = BayesianOptimization(
            lambda n_estimators, max_depth:
            bayesian_wrapper(
                ExtraTreesRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    n_jobs=-1),
                train_indices,
                input_data,
                target_data),
            param_set)

    elif strategy_dictionary['regression_mode'] == 'classification':
        bo = BayesianOptimization(
            lambda n_estimators, max_depth:
            bayesian_wrapper(
                ExtraTreesClassifier(
                    n_estimators=int(n_estimators),
                    max_depth = int(max_depth),
                    n_jobs=-1),
                train_indices,
                input_data,
                target_data),
            param_set)

    bo.maximize(n_iter=strategy_dictionary['ml_iterations'])

    result_dict = bo.res['max']

    error = 1 / result_dict['max_val']

    if strategy_dictionary['regression_mode'] == 'regression':
        model = GradientBoostingRegressor(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            max_depth=int(result_dict['max_params']['max_depth']))

    elif strategy_dictionary['regression_mode'] == 'classification':
        model = GradientBoostingClassifier(
            n_estimators=int(result_dict['max_params']['n_estimators']),
            max_depth=int(result_dict['max_params']['max_depth']))

    model.fit(input_data[train_indices, :], target_data[train_indices])

    return model, error


def tensorflow_fitting(train_indices, test_indices, validation_indices, input_data, target_data):

    """use keras/tensorflow to fit training inputs to targets and predict targets"""

    target_scaler = StandardScaler()
    target_data = target_scaler.fit_transform(target_data.reshape(-1, 1))

    input_data = input_data[:, :, None]

    train_data = input_data[train_indices, :, :]
    train_target = target_data[train_indices, None]

    test_data = input_data[test_indices, :, :]

    val_data = input_data[validation_indices, :, :]

    input_size = input_data.shape

    input1 = Input(shape=(input_size[1], input_size[2]))

    dropout1 = Dropout(0.2)(input1)
    dense1 = Dense(input_size[2], activation='tanh')(dropout1)
    preds = MaxPooling1D(2)(dense1)

    model = Model(input1, preds)

    model.compile(
        loss='mean_squared_error',
        optimizer='rmsprop')

    model.summary()

    model.fit(x=train_data, y=train_target)

    training_strategy_score = model.predict(train_data)
    fitted_strategy_score = model.predict(test_data)
    validation_strategy_score = model.predict(val_data)

    error = mean_squared_error(np.squeeze(train_target), np.squeeze(training_strategy_score))

    fitting_dictionary = {
        'training_strategy_score': training_strategy_score,
        'fitted_strategy_score': fitted_strategy_score,
        'validation_strategy_score': validation_strategy_score,
        'error': error,
    }

    return fitting_dictionary, error


def tensorflow_sequence_fitting(
        train_indices,
        test_indices,
        validation_indices,
        input_data,
        target_data):

    """use keras/tensorflow to fit training inputs to targets as sequence and predict targets"""

    input_data = input_data[:, :, None]

    target_scaler = StandardScaler()
    target_data = target_scaler.fit_transform(target_data.reshape(-1, 1))

    train_data = input_data[train_indices, :, :]
    train_target = target_data[train_indices, None]

    test_data = input_data[test_indices, :, :]

    val_data = input_data[validation_indices, :, :]

    input_size = input_data.shape

    model = Sequential()

    model.add(Dropout(0.2, input_shape=(input_size[1], input_size[2],)))
    model.add(Dense(2))
    model.add(MaxPooling1D(2))
    model.add(LSTM(1, dropout=0.2, recurrent_dropout=0.2))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.summary()

    training_strategy_score = model.predict(train_data)
    fitted_strategy_score = model.predict(test_data)
    validation_strategy_score = model.predict(val_data)

    error = mean_squared_error(np.squeeze(train_target), np.squeeze(training_strategy_score))

    fitting_dictionary = {
        'training_strategy_score': training_strategy_score,
        'fitted_strategy_score': fitted_strategy_score,
        'validation_strategy_score': validation_strategy_score,
        'error': error,
    }

    return fitting_dictionary, fitting_dictionary['error']
