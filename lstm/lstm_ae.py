from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, LSTM, Reshape
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras import models
from keras.models import Model
import keras
import numpy as np
from pdb import set_trace


learning_rate = 0.0001
epochs = 200

in_sample = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
])

n_seq = len(in_sample)
n_token = len(in_sample[0])
in_sample = in_sample.reshape((n_seq, n_token, 1))

# Define input shape
model = models.Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_token, 1)))
model.add(RepeatVector(n_token))
model.add(LSTM(200, activation='relu', return_sequences=True))
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
    sample = np.arange(start_num, start_num+0.9, 0.1)
    sample = sample.reshape((1, len(sample), 1))
    return sample


while True:
    try:
        start_num = float(input('input the start num >> '))
        sample = gen_sample(start_num)
        #ret = model.predict(np.random.normal(size=(1, 2, 2)))
        ret = model.predict(sample)
        print(f"predict: {ret}")
    except:
        print("over")
        exit(0)
