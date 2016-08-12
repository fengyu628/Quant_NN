# coding:utf-8

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
from data import get_data

input_dim = 8
input_length = 100

print('Build model...')
model = Sequential()
print('-'*50)
model.add(LSTM(20, dropout_W=0.2, dropout_U=0.2, input_dim=input_dim, input_length=input_length))
print('-'*50)
model.add(Dense(1))
print('-'*50)
# model.add(Activation('sigmoid'))
print(model.summary())

print('-'*50)
print('Compile...')
model.compile(loss='mean_squared_error',
              optimizer='SGD',
              metrics=['accuracy'])

print('-'*50)
train_data, target_data = get_data()
print('data shape:', train_data.shape, target_data.shape)
batch_size = 100
print('-'*50)
print('Train...')
model.fit(train_data, target_data, batch_size=batch_size, nb_epoch=10, verbose=2, validation_split=0.2, shuffle=False)
