# coding:utf-8
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QFileDialog
import numpy as np
import theano
import theano.tensor as T
# print(np.array(1))
# a = tensor.tensor3()
a = [1,2,3,4,5,6,7,8]
b = [2,4,6,8,10,12,14,16]

# x1_s = tensor.vector()
# x2_s = tensor.vector()
# result, update = theano.scan(lambda x1, x2:(x1+x2), sequences=[x1_s, x2_s])
# f = theano.function([x1_s, x2_s], result)
# c = f(a, b)
# print(c)


def loss_variance(y_out, y_target):
    return T.mean(T.sqr(y_out - y_target))

c = loss_variance(np.asarray(a), np.asarray(b))
print(c.eval())