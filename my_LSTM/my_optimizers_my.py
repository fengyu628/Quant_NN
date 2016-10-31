# coding:utf-8

import theano
import theano.tensor as T
import numpy as np


class Optimizer_SGD_my(object):

    def __init__(self, lr=0.1, momentum=0.5, decay=0.01, nesterov=False):
        super(Optimizer_SGD_my, self).__init__()
        self.lr = theano.shared(lr)
        self.momentum = theano.shared(momentum)
        self.decay = theano.shared(decay)
        self.nesterov = nesterov
        self.updates = []
        self.momentum_weights = None

    def get_updates(self, weights_list, grads_list, loss, constraints):
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


class Optimizer_Adagrad_my(object):

    def __init__(self, lr=0.01, epsilon=1e-6):
        super(Optimizer_Adagrad_my, self).__init__()
        self.lr = theano.shared(lr)
        self.epsilon = theano.shared(epsilon)
        self.accumulators = None
        self.updates = []

    def get_updates(self, weights_list, grads_list, loss, constraints):
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


class Optimizer_Adadelta_my(object):

    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6):
        super(Optimizer_Adadelta_my, self).__init__()
        self.lr = theano.shared(lr)
        self.rho = theano.shared(rho)
        self.epsilon = theano.shared(epsilon)
        self.weights = None
        self.updates = []

    def get_updates(self, weights_list, grads_list, loss, constraints):
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


class Optimizer_RMSprop_my(object):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6):
        super(Optimizer_RMSprop_my, self).__init__()
        self.lr = theano.shared(lr)
        self.rho = theano.shared(rho)
        self.epsilon = epsilon
        self.weights = None
        self.updates = []

    def get_updates(self, weights_list, grads_list, loss, constraints):
        # 计算梯度
        grads = T.grad(loss, weights_list)
        # accumulators
        self.weights = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]

        for p, g, a in zip(weights_list, grads, self.weights):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * T.sqr(g)
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            self.updates.append((p, new_p))

        # 取得梯度数据，用于显示或查看
        for gl, g in zip(grads_list, grads):
            self.updates.append((gl, g))

        return self.updates


class Optimizer_Adam_my(object):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(Optimizer_Adam_my, self).__init__()
        self.iterations = theano.shared(0.)
        self.lr = theano.shared(lr)
        self.beta_1 = theano.shared(beta_1)
        self.beta_2 = theano.shared(beta_2)
        self.epsilon = epsilon
        self.updates = []
        self.weights = None

    def get_updates(self, weights_list, grads_list, loss, constraints):
        # 计算梯度
        grads = T.grad(loss, weights_list)
        self.updates = [(self.iterations, self.iterations + 1.)]

        t = self.iterations + 1
        lr_t = self.lr * T.sqrt(1. - T.pow(self.beta_2, t)) / (1. - T.pow(self.beta_1, t))

        ms = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        vs = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        self.weights = ms + vs

        for p, g, m, v in zip(weights_list, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * T.sqr(g)
            p_t = p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)
            self.updates.append((m, m_t))
            self.updates.append((v, v_t))
            new_p = p_t
            self.updates.append((p, new_p))

        # 取得梯度数据，用于显示或查看
        for gl, g in zip(grads_list, grads):
            self.updates.append((gl, g))

        return self.updates


class Optimizer_Adamax_my(object):

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(Optimizer_Adamax_my, self).__init__()
        self.iterations = theano.shared(0.)
        self.lr = theano.shared(lr)
        self.beta_1 = theano.shared(beta_1)
        self.beta_2 = theano.shared(beta_2)
        self.epsilon = epsilon
        self.updates = []
        self.weights = None

    def get_updates(self, weights_list, grads_list, loss, constraints):
        # 计算梯度
        grads = T.grad(loss, weights_list)
        self.updates = [(self.iterations, self.iterations + 1.)]

        t = self.iterations + 1
        lr_t = self.lr / (1. - T.pow(self.beta_1, t))

        # zero init of 1st moment
        ms = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        # zero init of exponentially weighted infinity norm
        us = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        self.weights = ms + us

        for p, g, m, u in zip(weights_list, grads, ms, us):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = T.maximum(self.beta_2 * u, T.abs_(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)
            self.updates.append((m, m_t))
            self.updates.append((u, u_t))
            new_p = p_t
            self.updates.append((p, new_p))

        # 取得梯度数据，用于显示或查看
        for gl, g in zip(grads_list, grads):
            self.updates.append((gl, g))

        return self.updates


class Optimizer_Nadam_my(object):

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004):
        super(Optimizer_Nadam_my, self).__init__()
        self.iterations = theano.shared(0.)
        self.m_schedule = theano.shared(1.)
        self.lr = theano.shared(lr)
        self.beta_1 = theano.shared(beta_1)
        self.beta_2 = theano.shared(beta_2)
        self.schedule_decay = schedule_decay
        self.epsilon = epsilon
        self.updates = []
        self.weights = None

    def get_updates(self, weights_list, grads_list, loss, constraints):
        # 计算梯度
        grads = T.grad(loss, weights_list)
        # self.updates = [K.update_add(self.iterations, 1)]
        self.updates = [(self.iterations, self.iterations + 1.)]

        t = self.iterations + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (T.pow(0.96, t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (T.pow(0.96, (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        # shapes = [K.get_variable_shape(p) for p in weights_list]
        # ms = [K.zeros(shape) for shape in shapes]
        # vs = [K.zeros(shape) for shape in shapes]
        ms = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]
        vs = [theano.shared(np.zeros(p.get_value().shape)) for p in weights_list]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(weights_list, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * T.sqr(g)
            v_t_prime = v_t / (1. - T.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))

            p_t = p - self.lr * m_t_bar / (T.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t

            self.updates.append((p, new_p))

        # 取得梯度数据，用于显示或查看
        for gl, g in zip(grads_list, grads):
            self.updates.append((gl, g))

        return self.updates
