# coding:utf-8

import numpy as np
import theano.tensor as T


# 方差损失函数
def loss_variance(y_out, y_target):
    return T.mean(T.sqr(y_out - y_target))
    # return tensor.sqr(y - y_target).get_value()


def loss_other():
    pass

variance = loss_variance
