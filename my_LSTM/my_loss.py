# coding:utf-8

import numpy as np
import theano.tensor as T


# 方差损失函数
def loss_mean_squared_error(y_out, y_target):
    return T.mean(T.sqr(y_out - y_target))
    # return tensor.sqr(y - y_target).get_value()


def loss_mean_absolute_error(y_true, y_pred):
    return T.mean(T.abs_(y_pred - y_true), axis=-1)
