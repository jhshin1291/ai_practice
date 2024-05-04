from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, LSTM, Reshape
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras import models
from keras.models import Model
import keras
import numpy as np
from sklearn.metrics import mean_squared_error

import pdb
from pdb import set_trace


learning_rate = 0.001
epochs = 250

#in_sample = np.arange(0, 10, 0.1).reshape((10, 10))
#new_sample = np.arange(4, 5, 0.1)
#new_sample[4] = 10
#new_sample[9] = 10
#array([4. ], [4.1], [4.2], [4.3], [10.0], [4.4], [4.6], [4.7], [4.8], [10.0])

in_sample = np.random.normal(size=(10, 10))
print(in_sample)

'''
array([[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
       [1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
       [2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
       [3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],
       [4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9],
       [5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9],
       [6. , 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9],
       [7. , 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9],
       [8. , 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9],
       [9. , 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9]])
'''

n_seq = len(in_sample)
n_token = len(in_sample[0])
in_sample = in_sample.reshape((n_seq, n_token, 1))

# Define input shape
model = models.Sequential()
model.add(LSTM(20, activation='relu', return_sequences=True, input_shape=(n_token, 1)))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(200, activation='relu'))
model.add(RepeatVector(n_token))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(20, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

# Compile the model
opt = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='mse')

# Print model summary
model.summary()

# Train the model
model.fit(in_sample, in_sample,
                epochs=epochs,
                batch_size=64,
                shuffle=True)

def gen_sample(start_num):
    sample = np.arange(start_num, start_num+1.0, 0.1)
    sample = sample.reshape((1, len(sample), 1))
    return sample

def evaluation(pred, in_data):
    pred = np.reshape(pred, (10,))
    in_data = np.reshape(in_data, (10,))
    mse = mean_squared_error(in_data, pred)
    return mse

def pred1():
    start_num = float(input('input the start num >> '))
    in_test_sample = gen_sample(start_num)
    set_trace()
    #ret = model.predict(np.random.normal(size=(1, 2, 2)))
    pred = model.predict(in_test_sample)
    print(f"predict: {pred}")
    mse = evaluation(pred, in_test_sample)
    print(f"diff(mse): {mse}")

    if mse < 0.002:
        print("Result: Normal!")
    else:
        print("Result: Anomaly!")

def pred2():
    idx = int(input('which seq do you want to test? >> '))
    set_trace()
    in_test_sample = in_sample[idx]
    in_test_sample = in_test_sample.reshape((1, 10))
    print(f"Input: {in_test_sample}")
    pred = model.predict(in_test_sample)
    print(f"predict: {pred}")
    mse = evaluation(pred, in_test_sample)
    print(f"diff(mse): {mse}")

    if mse < 0.002:
        print("Result: Normal!")
    else:
        print("Result: Anomaly!")



while True:
    try:
        pred2()
    except Exception as e:
        print(e)
        print("over")
        exit(0)
