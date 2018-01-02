# This code is inspired from : http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html#bayesian-optimization
import numpy as np
from sklearn.gaussian_process import GaussianProcess
import random
import warnings


warnings.filterwarnings("ignore")

def vector_2d(array):
    # return np.array(array).reshape((-1, 1))
    return array

def gaussian_process(x_train, y_train, x_test):
    x_train = vector_2d(x_train)
    y_train = vector_2d(y_train)
    x_test = vector_2d(x_test)

    # Train gaussian process
    gp = GaussianProcess(corr='squared_exponential',
                         theta0=1e-1, thetaL=1e-3, thetaU=1)
    print (x_train)
    print (y_train)
    gp.fit(x_train, y_train)

    # Get mean and standard deviation for each possible
    # number of hidden units
    y_mean, y_var = gp.predict(x_test, eval_MSE=True)
    y_std = np.sqrt(vector_2d(y_var))

    return y_mean, y_std

def next_parameter_by_ei(y_min, y_mean, y_std, x_choices):
    # Calculate expecte improvement from 95% confidence interval
    expected_improvement = np.array(y_min) - (np.array(y_mean).flatten() - 1.96 * np.array(y_std))
    expected_improvement[expected_improvement < 0] = 0

    max_index = expected_improvement.argmax()
    # Select next choice
    next_parameter = x_choices[max_index]

    return next_parameter


def  randomly_select_params(parameters, n_choese, size_of_n_chose):
    repetitive_flag = 0
    if len(parameters) == len(n_choese):
        return 0
    else:
        while(True):
            params = n_choese[random.randint(0, (size_of_n_chose-1))]
            for i in range(len(parameters)):
                if (np.array_equal(params,parameters[i])):
                    repetitive_flag = 1
            if repetitive_flag == 0:
                return params

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
        # out = np.zeros([n, len(arrays)])

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:],
                  out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
def hyperparam_selection(func, parameters_ranges, func_args=None, n_iter=20):
    if func_args is None:
        func_args = []

    scores = []
    parameters = []

    # min_n_hidden, max_n_hidden, n_hidden_step = n_hidden_range
    # min_n_epoch, max_n_epoch, n_epoch_step = n_epoch_range
    parameters_choices = []
    len_of_comb_of_parameters = 1
    for i in range (len(parameters_ranges)):
        parameters_choices.append(np.arange(parameters_ranges[i][0], parameters_ranges[i][1]+1, parameters_ranges[i][2]) )
        len_of_comb_of_parameters *= len(parameters_choices[i])

    # n_epoch_choices = np.arange(min_n_epoch, max_n_epoch + 1, n_epoch_step)
    # min_choeses = np.min([len(n_hidden_choices), len(n_epoch_choices)])
    n_choese = cartesian (parameters_choices)
    # for k in range (len(parameters_ranges)):
    #     for i in range (len(parameters_ranges)):
    #         for j in range(0, len(parameters_choices)):
    #         n_choese.append ()
    #         n_choese.append([n_hidden_choices[i], n_epoch_choices[j]])

    # To be able to perform gaussian process we need to
    # have at least 2 samples.
    # n_hidden = n_hidden_choices[random.randint(0, (max_n_hidden - min_n_hidden) / n_hidden_step - 1)]
    # n_epoch = n_epoch_choices[random.randint(0, (max_n_epoch - min_n_epoch) / n_epoch_step - 1)]

    params = n_choese[random.randint(0,len_of_comb_of_parameters)]
    # n_hidden, n_epoch = params

    score = func(params, *func_args)

    parameters.append(params)
    # parameters.append(n_hidden)
    scores.append([score])

    # n_hidden = n_hidden_choices[random.randint(0, (max_n_hidden - min_n_hidden)/n_hidden_step -1)]
    # n_epoch = n_epoch_choices[random.randint(0, (max_n_epoch - min_n_epoch) / n_epoch_step - 1)]
    params = randomly_select_params(parameters, n_choese, len_of_comb_of_parameters)
    # params = n_choese[random.randint(0, i*j)]
    # n_hidden, n_epoch = params
    for iteration in range(2, n_iter + 1):

        # n_hidden, n_epoch = params
        # score = func(n_hidden, *func_args)
        score = func(params, *func_args)
        print ("score (MSE) for the called model is:" + str(score))

        # parameters.append(n_hidden)
        parameters.append(params)
        print ("params are:" + str(params))
        scores.append([score])

        y_min = min(scores)
        y_mean, y_std = gaussian_process(parameters, scores,
                                         n_choese)
                                         # n_hidden_choices)

        params = next_parameter_by_ei(y_min, y_mean, y_std,
                                        n_choese)
                                        # n_hidden_choices)

        if y_min == 0:
            # Lowest expected improvement value have been achieved
            break

    min_score_index = np.argmin(scores)
    return parameters[min_score_index]