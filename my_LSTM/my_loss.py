# coding:utf-8

import numpy as np


# 方差损失函数
def loss_variance(y, y_target):
    return np.square(y - y_target)  # .mean()
    # return tensor.sqr(y - y_target).get_value()


def loss_other():
    pass

variance = loss_variance
