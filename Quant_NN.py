#coding:utf-8

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
from data import get_data

input_dim = 8
input_length=32

print('Build model...')
model = Sequential()
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, input_dim=input_dim, input_length=input_length))
model.add(Dense(1))
# model.add(Activation('sigmoid'))
print(model.summary())

print('Compile...')
model.compile(loss='mean_squared_error',
              optimizer='SGD',
              metrics=['accuracy'])

train_data, target_data = get_data()
print('data shape:', np.asarray(train_data).shape)
batch_size = 100
print('Train...')
model.fit(train_data, target_data, batch_size=batch_size, nb_epoch=10, verbose=2, validation_split=0.2, shuffle=False)