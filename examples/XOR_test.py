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
model.add(Dense(2, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

print 'compiling...'
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

print 'fitting...'
history = model.fit(X, y, nb_epoch=1000, batch_size=4, verbose=1)

print 'evaluate'
print model.evaluate(X,y)
print 'predict'
print model.predict_classes(X)