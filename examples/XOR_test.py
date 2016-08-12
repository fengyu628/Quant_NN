#coding:utf-8

import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

X = np.zeros((4, 2), dtype='uint8')
y = np.zeros(4, dtype='uint8')

X[0] = [0, 0]
y[0] = 0
X[1] = [0, 1]
y[1] = 1
X[2] = [1, 0]
y[2] = 1
X[3] = [1, 1]
y[3] = 0

print 'making model...'
model = Sequential()
print('-'*50)
model.add(Dense(2, input_dim=2))
print('-'*50)
model.add(Activation('tanh'))
print('-'*50)
model.add(Dense(1))
print('-'*50)
model.add(Activation('sigmoid'))
print('-'*50)
print model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

print('-'*50)
print 'compiling...'
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

print('-'*50)
print 'fitting...'
history = model.fit(X, y, nb_epoch=10, batch_size=4, verbose=1)
print history

print('-'*50)
print 'evaluate...'
print model.evaluate(X,y)
print('-'*50)
print 'predict...'
print model.predict_classes(X)