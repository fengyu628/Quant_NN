# coding:utf-8

import numpy as np
import theano_backend as K


class Layer(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = None
        self.output = None


class Dense(Layer):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        # 初始化权值
        self.weight_W = np.zeros((self.input_dim, self.output_dim))
        self.weight_b = np.zeros(self.output_dim)

    def call(self, x):
        return np.dot(self.weight_W, self.input) + self.weight_b

    def build(self):
        pass


class ActivationTanH (Layer):
    def __init__(self, *args, **kwargs):
        super(ActivationTanH, self).__init__(*args, **kwargs)

    def call(self, x):
        return K.tanh(x)

    def build(self):
        pass
