# coding:utf-8

import numpy as np
import theano
import theano.tensor as tensor
from theano import config

input_dim = 2
inner_units = 100

data = np.loadtxt('data\\sin.txt')
n_steps = len(data)
print('n_steps:', n_steps)

# 生成权值矩阵，shape为[input_dim, inner_units]
def make_matrix_InputDim_InnerUnits():
    randn = np.random.rand(input_dim, inner_units)
    W = (0.01 * randn).astype(config.floatX)
    W = theano.shared(W)
    return W

# 生成权值矩阵，shape为[inner_units, inner_units]
def make_matrix_InnerUnits_InnerUnits():
    randn = np.random.rand(inner_units, inner_units)
    R = (0.01 * randn).astype(config.floatX)
    R = theano.shared(R)
    return R

# 生成权值向量，shape为[inner_units,]
def make_vector_InnerUnits():
    randn = np.random.random(inner_units)
    b = (0.01 * randn).astype(config.floatX)
    b = theano.shared(b)
    return b

def make_scalar():
    rand = np.random.random()
    s = (0.01 * rand)#.astype(config.floatX)
    s = theano.shared(s)
    return s


# 生成权值
W_i = make_matrix_InputDim_InnerUnits()
W_o = make_matrix_InputDim_InnerUnits()
W_f = make_matrix_InputDim_InnerUnits()
W_z = make_matrix_InputDim_InnerUnits()

R_i = make_matrix_InnerUnits_InnerUnits()
R_o = make_matrix_InnerUnits_InnerUnits()
R_f = make_matrix_InnerUnits_InnerUnits()
R_z = make_matrix_InnerUnits_InnerUnits()

b_i = make_vector_InnerUnits()
b_o = make_vector_InnerUnits()
b_f = make_vector_InnerUnits()
b_z = make_vector_InnerUnits()

p_i = make_vector_InnerUnits()
p_o = make_vector_InnerUnits()
p_f = make_vector_InnerUnits()

U_y = make_vector_InnerUnits()
b_y = make_scalar()


def _step(x, h_, c_):
    # block的输入
    net_z = tensor.dot(h_, R_z) + tensor.dot(x, W_z) + b_z
    z = tensor.tanh(net_z)

    # 输入们的输出
    # net_i = tensor.dot(h_, R_i) + tensor.dot(x, W_i) + c_*p_i + b_i
    net_i = tensor.dot(h_, R_i) + tensor.dot(x, W_i) + b_i
    i = tensor.nnet.sigmoid(net_i)

    # 忘记门的输出
    # net_f = tensor.dot(h_, R_f) + tensor.dot(x, W_f) + c_*p_f + b_f
    net_f = tensor.dot(h_, R_f) + tensor.dot(x, W_f) + b_f
    f = tensor.nnet.sigmoid(net_f)

    # c_：上一个cell的状态， 得到的c为cell的输出
    c = f * c_ + i * z

    # 输出们的输出
    # net_o = tensor.dot(h_, R_o) + tensor.dot(x, W_o) + c*p_o + b_o
    net_o = tensor.dot(h_, R_o) + tensor.dot(x, W_o) + b_o
    o = tensor.nnet.sigmoid(net_o)

    # h:block的输出
    h = o * tensor.tanh(c)

    return h, c

scan_sequences_input = tensor.matrix(name='scan_input')
scan_out, scan_updates = theano.scan(_step,
                            sequences=[scan_sequences_input],
                            outputs_info=[tensor.alloc(np.asarray([0.], dtype=config.floatX),
                                                       # n_samples,
                                                       inner_units),
                                          tensor.alloc(np.asarray([0.], dtype=config.floatX),
                                                       # n_samples,
                                                       inner_units)],
                            # name=_p(prefix, '_layers'),
                            # n_steps=n_steps
                            )
# scan_out[0]为block的输出h
# print(np.asarray(scan_out[0]).shape)
# U*y + b ， 最终的输出
out_y = tensor.dot(scan_out[0], U_y) + b_y
# out_softmax = tensor.nnet.softmax(tensor.dot(scan_out[0], U_y) + b_y)
# 制作输出函数
f_out = theano.function([scan_sequences_input], out_y, name='f_out')

a = f_out(data)
print(np.asarray(a).shape)
print(a)

cost = np.square(out_y - y)