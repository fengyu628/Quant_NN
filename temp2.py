# coding:utf-8
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QFileDialog
import numpy as np
import theano
import theano.tensor as tensor
print(np.array(1))
a = tensor.tensor3()
# a = [[[1,2],[3,4]],[[5,6],[7,8]]]


print(a.dim)