# coding:utf-8

import theano
import theano.tensor as T
import numpy as np


class Optimizer_SGD(object):
    def __init__(self, lr=0.1, momentum=0.5, decay=0.01, nesterov=False):
        super(Optimizer_SGD, self).__init__()
        self.lr = theano.shared(lr)
        self.momentum = theano.shared(momentum)
        self.decay = theano.shared(decay)
        self.nesterov = nesterov
        self.updates = []
        self.momentum_weights = None

    def make_updates(self, weights_list, grads_list, loss):
        # 计算梯度
        grads = T.grad(loss, weights_list)
        # 记录权值更新的动量
        self.momentum_weights = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        for p, g, m in zip(weights_list, grads, self.momentum_weights):
            v = self.momentum * m - self.lr * g
            # 更新动量
            self.updates.append((m, v))
            # 更新权值，没有验证此处的 nesterov 跟理论算法一致
            if self.nesterov:
                new_p = p + self.momentum * v - self.lr * g
            else:
                new_p = p + v
            self.updates.append((p, new_p))

        # 取得梯度数据，用于显示或查看
        for gl, g in zip(grads_list, grads):
            self.updates.append((gl, g))

        return self.updates

    def update_lr(self):
        self.lr /= (1 + self.decay)


class Optimizer_Adagrad(object):
    def __init__(self, lr=0.01, epsilon=1e-6):
        super(Optimizer_Adagrad, self).__init__()
        self.lr = theano.shared(lr)
        self.epsilon = theano.shared(epsilon)
        self.accumulators = None
        self.updates = []

    def make_updates(self, weights_list, grads_list, loss):
        # 计算梯度
        grads = T.grad(loss, weights_list)
        # accumulators
        self.accumulators = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]

        for p, g, a in zip(weights_list, grads, self.accumulators):
            new_a = a + T.sqr(g)  # update accumulator
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            self.updates.append((p, new_p))

        # 取得梯度数据，用于显示或查看
        for gl, g in zip(grads_list, grads):
            self.updates.append((gl, g))

        return self.updates


class Optimizer_Adadelta(object):
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6):
        super(Optimizer_Adadelta, self).__init__()
        self.lr = theano.shared(lr)
        self.rho = theano.shared(rho)
        self.epsilon = theano.shared(epsilon)
        self.weights = None
        self.updates = []

    def make_updates(self, weights_list, grads_list, loss):
        # 计算梯度
        grads = T.grad(loss, weights_list)
        accumulators = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        delta_accumulators = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        self.weights = accumulators + delta_accumulators

        for p, g, a, d_a in zip(weights_list, grads, accumulators, delta_accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * T.sqr(g)
            self.updates.append((a, new_a))
            # use the new accumulator and the *old* delta_accumulator
            update = g * T.sqrt(d_a + self.epsilon) / T.sqrt(new_a + self.epsilon)
            new_p = p - self.lr * update
            self.updates.append((p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * T.sqr(update)
            self.updates.append((d_a, new_d_a))

        # 取得梯度数据，用于显示或查看
        for gl, g in zip(grads_list, grads):
            self.updates.append((gl, g))

        return self.updates
