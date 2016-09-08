# coding:utf-8

import numpy as np
import theano
import theano.tensor as tensor
from theano import config

input_dim = 2
inner_units = 100
lr = 0.001
epoch = 100

data = np.loadtxt('data\\sin.txt')
n_steps = len(data)
print('n_steps:', n_steps)



# 生成随机权值矩阵
def make_random_matrix_with_shape(dim1, dim2):
    randn = np.random.rand(dim1, dim2)
    m = (0.01 * randn).astype(config.floatX)
    m = theano.shared(m)
    return m


# 生成随机权值向量
def make_random_vector_with_shape(dim1):
    randn = np.random.random(dim1)
    v = (0.01 * randn).astype(config.floatX)
    v = theano.shared(v)
    return v


# 生成随机标量
def make_scalar():
    rand = np.random.random()
    s = (0.01 * rand)#.astype(config.floatX)
    s = theano.shared(s)
    return s


# 生成权值
print('make weight')
W_i = make_random_matrix_with_shape(input_dim, inner_units)
W_o = make_random_matrix_with_shape(input_dim, inner_units)
W_f = make_random_matrix_with_shape(input_dim, inner_units)
W_z = make_random_matrix_with_shape(input_dim, inner_units)

R_i = make_random_matrix_with_shape(inner_units, inner_units)
R_o = make_random_matrix_with_shape(inner_units, inner_units)
R_f = make_random_matrix_with_shape(inner_units, inner_units)
R_z = make_random_matrix_with_shape(inner_units, inner_units)

b_i = make_random_vector_with_shape(inner_units)
b_o = make_random_vector_with_shape(inner_units)
b_f = make_random_vector_with_shape(inner_units)
b_z = make_random_vector_with_shape(inner_units)

p_i = make_random_vector_with_shape(inner_units)
p_o = make_random_vector_with_shape(inner_units)
p_f = make_random_vector_with_shape(inner_units)

U_y = make_random_vector_with_shape(inner_units)
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

x_symbol = tensor.matrix(name='scan_input')
scan_out, scan_updates = theano.scan(_step,
                            sequences=[x_symbol],
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
y_out_symbol = tensor.dot(scan_out[0], U_y) + b_y
# out_softmax = tensor.nnet.softmax(tensor.dot(scan_out[0], U_y) + b_y)
# 制作输出函数
f_out = theano.function([x_symbol], y_out_symbol,
                        name='f_out')

# a = f_out(data)
# print(np.asarray(a).shape)
# print(a)

# 目标值
y_target_symbol = tensor.scalar(name='y')

cost_symbol = np.square(y_out_symbol - y_target_symbol).mean()

# 计算梯度
# weights_list = [W_i, W_o, W_f, W_z, R_i, R_o, R_f, R_z, b_i, b_o, b_f, b_z, p_i, p_o, p_f, U_y, b_y]
weights_list = [W_i, W_o, W_f, W_z, R_i, R_o, R_f, R_z, b_i, b_o, b_f, b_z, U_y, b_y]

grads = tensor.grad(cost_symbol, wrt=weights_list)

grads_shared = [theano.shared(p.get_value() * 0.) for  p in weights_list]
grads_update = [(gs, g) for gs, g in zip(grads_shared, grads)]
# 制作损失函数
print('make cost function')
function_cost = theano.function([x_symbol, y_target_symbol], cost_symbol,
                                updates=grads_update,
                                name='sgd_f_grad_shared')

# 学习率
lr_symbol = tensor.scalar(name='learning rate')
weights_update = [(p, p - lr_symbol * g) for p, g in zip(weights_list, grads_shared)]
# 制作权值更新函数
print('make update function')
function_update = theano.function([lr_symbol], [],
                                  updates=weights_update,
                                  name='sgd_f_update')

x_length = 20
train_x_list = []
train_y_list = []
train_list_length = 0
for i in range(len(data) - x_length - 1):
    train_x_list.append(data[i:i+x_length, ])
    train_y_list.append(data[i+x_length, 0])
    train_list_length += 1
# print(train_x)
# print(train_y)

for epoch_index in range(epoch):
    print('epoch: %d' % epoch_index)

    for train_index in range(train_list_length):
        # print('train_index: ', train_index)

        x = train_x_list[train_index]
        y = train_y_list[train_index]

        cost = function_cost(x, y)
        if train_index % 20 == 0:
            # print('train_index: %d' % train_index)
            print('cost: %f' % cost)

        function_update(lr)