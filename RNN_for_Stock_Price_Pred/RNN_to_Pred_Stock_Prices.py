# This is code is inspired from : https://www.youtube.com/watch?v=ftMq5ps503w

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries

look_before_window_size = 100
look_ahead_window_size = 1
#Step 1 Load Data
X_train, y_train, X_test, y_test = lstm.load_data('NVIDIA.csv', look_before_window_size, True)

#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

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
    batch_size=16,
    nb_epoch=20,
    validation_split=0.05)


#Step 4 - Plot the predictions!
predictions = lstm.predict_sequences_multiple(model, X_test, look_before_window_size, look_ahead_window_size)
lstm.plot_results_multiple(predictions, y_test, look_before_window_size)
