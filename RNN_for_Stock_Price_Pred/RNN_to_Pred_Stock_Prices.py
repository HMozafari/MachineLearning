# This is code is inspired from : https://www.youtube.com/watch?v=ftMq5ps503w

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries
from Bayssian_Opt import hyperparam_selection

look_before_window_size = 100
look_ahead_window_size = 2
#Step 1 Load Data



def train_model (params,  X_train, y_train, X_test, y_test, dropOut=0.2, batch_size=64):
    #Step 2 Build Model
    DroputRate =float(params[3])/10.0  # Dropout
    model = Sequential()

    model.add(LSTM(
        input_dim=1,
        output_dim=params[0],
        return_sequences=True))
    model.add(Dropout(DroputRate))

    model.add(LSTM(
        2*params[0],
        return_sequences=False))
    model.add(Dropout(DroputRate))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='adam')
    print ('compilation time :', time.time() - start)

    #Step 3 Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=params[2],
        nb_epoch=params[1],
        validation_split=0.05)

    # Step 4 - Plot the predictions!
    predictions = lstm.predict_sequences_multiple(model, X_test, look_before_window_size, look_ahead_window_size)
    MSE = lstm.mean_square_error(predictions, y_test)
    # lstm.plot_results_multiple(predictions, y_test, look_before_window_size)
    return MSE

X_train, y_train, X_test, y_test = lstm.load_data('NVIDIA.csv', look_before_window_size, True)
parameters_ranges = []
parameters_ranges.append([100, 500, 50]) # [min, max, step] hidden units
parameters_ranges.append([1, 20, 2]) # [min, max, step] epochs
parameters_ranges.append([16, 128, 16]) # [min, max, step] batch_size
parameters_ranges.append([1, 5, 1]) # [min, max, step] 10 x dropout

best_n_hidden = hyperparam_selection(
    train_model,
    parameters_ranges,
    func_args=[X_train, y_train, X_test, y_test],
    n_iter=60,
)

print ("best number of hidden units is: " + str(best_n_hidden))