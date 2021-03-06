# coding:utf-8

import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from my_regularizers import *
from my_initializations import *


# 生成随机权值矩阵
def make_random_matrix_with_shape(dim1, dim2, name=None):
    # np.random.randn 为为0方差为1的正态分布
    randn = np.random.rand(dim1, dim2)
    # m = (0.01 * randn).astype(config.floatX)
    m = (0.02 * randn).astype(config.floatX)
    m = theano.shared(m, name=name)
    return m


# 生成随机权值向量
def make_random_vector_with_shape(dim1, name=None):
    randn = np.random.random(dim1)
    # v = (0.01 * randn).astype(config.floatX)
    v = (0.02 * randn).astype(config.floatX)
    v = theano.shared(v, name=name)
    return v


# 生成随机标量
def make_scalar(name=None):
    rand = np.random.random()
    # s = (0.01 * rand)
    s = (0.02 * rand)
    s = theano.shared(s, name=name)
    return s


class Layer_LSTM(object):

    def __init__(self, input_dim, inner_units):
        self.input_dim = input_dim
        self.inner_units = inner_units
        self.default_initializer = normal
        self.default_matrix_initializer = normal
        self.l1_default = 0.
        self.l2_default = 0.01
        # 生成权值
        print('make weight')
        self.W_i = self.default_matrix_initializer((input_dim, inner_units), name='InputGate Input Weight')
        self.W_o = self.default_matrix_initializer((input_dim, inner_units), name='OutputGate Input Weight')
        self.W_f = self.default_matrix_initializer((input_dim, inner_units), name='ForgetGate Input Weight')
        self.W_z = self.default_matrix_initializer((input_dim, inner_units), name='BlockInput Input Weight')
        self.W_i_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        self.W_o_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        self.W_f_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        self.W_z_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)

        self.R_i = self.default_matrix_initializer((inner_units, inner_units), name='InputGate Recurrent Weight')
        self.R_o = self.default_matrix_initializer((inner_units, inner_units), name='OutputGate Recurrent Weight')
        self.R_f = self.default_matrix_initializer((inner_units, inner_units), name='ForgetGate Recurrent Weight')
        self.R_z = self.default_matrix_initializer((inner_units, inner_units), name='BlockInput Recurrent Weight')
        self.R_i_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        self.R_o_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        self.R_f_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        self.R_z_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)

        self.b_i = self.default_initializer(inner_units, name='InputGate Bias')
        self.b_o = self.default_initializer(inner_units, name='OutputGate Bias')
        self.b_f = self.default_initializer(inner_units, name='ForgetGate Bias')
        self.b_z = self.default_initializer(inner_units, name='BlockInput Bias')
        # self.b_i_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        # self.b_o_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        # self.b_f_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)
        # self.b_z_regularizer = WeightRegularizer(l1=self.l1_default, l2=self.l2_default)

        # self.p_i = make_random_vector_with_shape(inner_units)
        # self.p_o = make_random_vector_with_shape(inner_units)
        # self.p_f = make_random_vector_with_shape(inner_units)

        self.U_y = self.default_initializer(inner_units, name='BlockOutput Weight')
        self.b_y = self.default_initializer(None, name='BlockOutput Bias')
        self.U_y_regularizer = WeightRegularizer(l1=0., l2=0.)
        # self.b_y_regularizer = WeightRegularizer(l1=0., l2=0.)

        # self.weights_list = [self.W_i, self.W_o, self.W_f, self.W_z,
        #                      self.R_i, self.R_o, self.R_f, self.R_z,
        #                      self.b_i, self.b_o, self.b_f, self.b_z,
        #                      self.p_i, self.p_o, self.p_f,
        #                      self.U_y, self.b_y]

        self.weights_list = [self.W_i, self.W_o, self.W_f, self.W_z,
                             self.R_i, self.R_o, self.R_f, self.R_z,
                             self.b_i, self.b_o, self.b_f, self.b_z,
                             self.U_y, self.b_y]

        self.W_i_regularizer.set_param(self.W_i)
        self.W_o_regularizer.set_param(self.W_o)
        self.W_f_regularizer.set_param(self.W_f)
        self.W_z_regularizer.set_param(self.W_z)

        self.R_i_regularizer.set_param(self.R_i)
        self.R_o_regularizer.set_param(self.R_o)
        self.R_f_regularizer.set_param(self.R_f)
        self.R_z_regularizer.set_param(self.R_z)

        # self.b_i_regularizer.set_param(self.b_i)
        # self.b_o_regularizer.set_param(self.b_o)
        # self.b_f_regularizer.set_param(self.b_f)
        # self.b_z_regularizer.set_param(self.b_z)

        self.U_y_regularizer.set_param(self.U_y)
        # self.b_y_regularizer.set_param(self.b_y)

        self.regularizers = []
        self.regularizers.append(self.W_i_regularizer)
        self.regularizers.append(self.W_o_regularizer)
        self.regularizers.append(self.W_f_regularizer)
        self.regularizers.append(self.W_z_regularizer)

        self.regularizers.append(self.R_i_regularizer)
        self.regularizers.append(self.R_o_regularizer)
        self.regularizers.append(self.R_i_regularizer)
        self.regularizers.append(self.R_i_regularizer)

        # self.regularizers.append(self.b_i_regularizer)
        # self.regularizers.append(self.b_i_regularizer)
        # self.regularizers.append(self.b_i_regularizer)
        # self.regularizers.append(self.b_i_regularizer)

        self.regularizers.append(self.U_y_regularizer)
        # self.regularizers.append(self.b_y_regularizer)

    def get_weight_list(self):
        return self.weights_list

    def _step(self, x_in, h_, c_):
        # block的输入
        net_z = tensor.dot(h_, self.R_z) + tensor.dot(x_in, self.W_z) + self.b_z
        z = tensor.tanh(net_z)

        # 输入们的输出
        # net_i = tensor.dot(h_, R_i) + tensor.dot(x_in, W_i) + c_*p_i + b_i
        net_i = tensor.dot(h_, self.R_i) + tensor.dot(x_in, self.W_i) + self.b_i
        i = tensor.nnet.sigmoid(net_i)

        # 忘记门的输出
        # net_f = tensor.dot(h_, R_f) + tensor.dot(x_in, W_f) + c_*p_f + b_f
        net_f = tensor.dot(h_, self.R_f) + tensor.dot(x_in, self.W_f) + self.b_f
        f = tensor.nnet.sigmoid(net_f)

        # c_：上一个cell的状态， 得到的c为cell的输出
        c = f * c_ + i * z

        # 输出们的输出
        # net_o = tensor.dot(h_, R_o) + tensor.dot(x_in, W_o) + c*p_o + b_o
        net_o = tensor.dot(h_, self.R_o) + tensor.dot(x_in, self.W_o) + self.b_o
        o = tensor.nnet.sigmoid(net_o)

        # h:block的输出
        h = o * tensor.tanh(c)

        return h, c

    def __call__(self, layer_input):
        scan_out, scan_updates = theano.scan(self._step,
                                             sequences=[layer_input],
                                             outputs_info=[tensor.alloc(np.asarray([0.], dtype=config.floatX),
                                                                        # n_samples,
                                                                        self.inner_units),
                                                           tensor.alloc(np.asarray([0.], dtype=config.floatX),
                                                                        # n_samples,
                                                                        self.inner_units)],
                                             # name=_p(prefix, '_layers'),
                                             # n_steps=n_steps
                                             )
        # scan_out[0]为block的输出h
        # print(np.asarray(scan_out[0]).shape)

        # 最终的输出（U*y + b）, 只取最后一个h，所以是scan_out[0][-1]
        layer_output = tensor.dot(scan_out[0][-1], self.U_y) + self.b_y
        # out_softmax = tensor.nnet.softmax(tensor.dot(scan_out[0][-1], U_y) + b_y)
        return layer_output


class Layer_GRU(object):

    def __init__(self):
        pass

lstm = Layer_LSTM
gru = Layer_GRU
