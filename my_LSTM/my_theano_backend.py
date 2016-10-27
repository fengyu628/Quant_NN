# coding:utf-8

import numpy as np
import theano
from theano import tensor as T

_FLOATX = theano.config.floatX
_LEARNING_PHASE = 0

def variable(value, dtype=_FLOATX, name=None):
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)


def switch(condition, then_expression, else_expression):
    return T.switch(condition, then_expression, else_expression)


def get_value(x):
    if not hasattr(x, 'get_value'):
        raise Exception("'get_value() can only be called on a variable. " +
                        "If you have an expression instead, use eval().")
    return x.get_value()


def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


def gradients(loss, variables):
    return T.grad(loss, variables)


def sqrt(x):
    x = T.clip(x, 0., np.inf)
    return T.sqrt(x)


def sum(x, axis=None, keepdims=False):
    return T.sum(x, axis=axis, keepdims=keepdims)


def square(x):
    return T.sqr(x)


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return T.clip(x, min_value, max_value)


def batch_get_value(xs):
    return [get_value(x) for x in xs]


def batch_set_value(tuples):
    for x, value in tuples:
        x.set_value(np.asarray(value, dtype=x.dtype))


def update_add(x, increment):
    return (x, x + increment)


def get_variable_shape(x):
    return x.get_value(borrow=True, return_internal_type=True).shape


def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


def update(x, new_x):
    return (x, new_x)


def pow(x, a):
    return T.pow(x, a)


def maximum(x, y):
    return T.maximum(x, y)


def abs(x):
    return T.abs_(x)


def cast_to_floatx(x):
    '''Cast a Numpy array to floatx.
    '''
    return np.asarray(x, dtype=_FLOATX)

def in_train_phase(x, alt):
    if _LEARNING_PHASE is 1:
        return x
    elif _LEARNING_PHASE is 0:
        return alt
    x = T.switch(_LEARNING_PHASE, x, alt)
    x._uses_learning_phase = True
    return x