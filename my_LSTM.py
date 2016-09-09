# coding:utf-8

import numpy as np
import theano
import theano.tensor as tensor
from theano import config

input_dim = 2
inner_units = 5
learning_rate = 0.001
epoch = 100

file_train_data = 'data\\sin.txt'
file_valid_data = 'data\\sin2.txt'
file_weights_saved = 'data\\LSTM_weights'

data_train = np.loadtxt(file_train_data)
data_valid = np.loadtxt(file_valid_data)

n_steps = len(data_train)
print('n_steps: %d' % n_steps)


# ======================================================================================================================
# ======================================================================================================================


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
    s = (0.01 * rand)
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

# weights_list = [W_i, W_o, W_f, W_z, R_i, R_o, R_f, R_z, b_i, b_o, b_f, b_z, p_i, p_o, p_f, U_y, b_y]
weights_list = [W_i, W_o, W_f, W_z, R_i, R_o, R_f, R_z, b_i, b_o, b_f, b_z, U_y, b_y]


# ======================================================================================================================
# ======================================================================================================================


def lstm_layer(layer_input):

    def _step(x_in, h_, c_):
        # block的输入
        net_z = tensor.dot(h_, R_z) + tensor.dot(x_in, W_z) + b_z
        z = tensor.tanh(net_z)

        # 输入们的输出
        # net_i = tensor.dot(h_, R_i) + tensor.dot(x_in, W_i) + c_*p_i + b_i
        net_i = tensor.dot(h_, R_i) + tensor.dot(x_in, W_i) + b_i
        i = tensor.nnet.sigmoid(net_i)

        # 忘记门的输出
        # net_f = tensor.dot(h_, R_f) + tensor.dot(x_in, W_f) + c_*p_f + b_f
        net_f = tensor.dot(h_, R_f) + tensor.dot(x_in, W_f) + b_f
        f = tensor.nnet.sigmoid(net_f)

        # c_：上一个cell的状态， 得到的c为cell的输出
        c = f * c_ + i * z

        # 输出们的输出
        # net_o = tensor.dot(h_, R_o) + tensor.dot(x_in, W_o) + c*p_o + b_o
        net_o = tensor.dot(h_, R_o) + tensor.dot(x_in, W_o) + b_o
        o = tensor.nnet.sigmoid(net_o)

        # h:block的输出
        h = o * tensor.tanh(c)

        return h, c

    scan_out, scan_updates = theano.scan(_step,
                                         sequences=[layer_input],
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

    # 最终的输出（U*y + b）, 只取最后一个h，所以是scan_out[0][-1]
    layer_output = tensor.dot(scan_out[0][-1], U_y) + b_y
    # out_softmax = tensor.nnet.softmax(tensor.dot(scan_out[0][-1], U_y) + b_y)
    return layer_output


# 制作layer输出函数
def make_function_layer_output(layer):
    x_symbol = tensor.matrix(name='scan_input')
    print('make output function')
    return theano.function([x_symbol], layer(x_symbol), name='f_out')


# 方差损失函数
def loss_variance(y, y_target):
    return np.square(y - y_target)  # .mean()


# 随机梯度下降
def sgd(layer, loss_function):

    x_symbol = tensor.matrix(name='scan_input')
    y_out_symbol = layer(x_symbol)

    # 目标值
    y_target_symbol = tensor.scalar(name='y')
    loss_symbol = loss_function(y_out_symbol, y_target_symbol)

    # 计算梯度
    grads = tensor.grad(loss_symbol, wrt=weights_list)

    grads_shared = [theano.shared(p.get_value() * 0.) for p in weights_list]
    grads_update = [(gs, g) for gs, g in zip(grads_shared, grads)]
    # 制作损失函数
    print('make loss function')
    f_loss = theano.function([x_symbol, y_target_symbol], loss_symbol,
                             updates=grads_update,
                             name='sgd_f_grad_shared')

    # 学习率
    lr_symbol = tensor.scalar(name='learning rate')
    weights_update = [(p, p - lr_symbol*g) for p, g in zip(weights_list, grads_shared)]
    # 制作权值更新函数
    print('make update function')
    f_update = theano.function([lr_symbol], [],
                               updates=weights_update,
                               name='sgd_f_update')

    return f_loss, f_update


# 把数据切成用于训练的小段
def slice_data(data_to_slice, slice_length):
    list_length = 0
    x_list = []
    y_list = []
    for i in range(len(data_to_slice) - slice_length - 6):
        x_list.append(data_to_slice[i:i+slice_length, ])
        y_list.append(data_to_slice[i+slice_length+5, 0])
        list_length += 1
    return x_list, y_list, list_length


# 计算验证误差
def error_valid(x_lise, y_list, list_length, layer_compute_function, loss_function):
    error = 0.
    for i in range(list_length):
        x = x_lise[i]
        y = y_list[i]
        out = layer_compute_function(x)
        loss_valid = loss_function(out, y)
        error += loss_valid
    error /= float(list_length)
    return error

# ======================================================================================================================
# ======================================================================================================================

# 生成layer输出函数
function_layer_output = make_function_layer_output(lstm_layer)
# 生成损失计算函数和权值更新函数
function_compute_loss, function_update_weights = sgd(lstm_layer, loss_variance)

# 生成测试数据
x_length = 20
x_train_list, y_train_list, train_list_length = slice_data(data_train, x_length)
x_valid_lise, y_valid_list, valid_list_length = slice_data(data_valid, x_length)

# ======================================================================================================================
# ======================================================================================================================

# 开始训练
for epoch_index in range(epoch):
    print('========== epoch: %d ==========' % epoch_index)

    for train_index in range(train_list_length):
        # print('train_index: ', train_index)

        x_train = x_train_list[train_index]
        y_train = y_train_list[train_index]

        # 计算目标函数
        loss = function_compute_loss(x_train, y_train)
        # if train_index % 20 == 0:
        #     print('loss: %f' % loss)

        # 更新权值
        function_update_weights(learning_rate)

    # 计算验证误差
    err = error_valid(x_valid_lise, y_valid_list, valid_list_length, function_layer_output, loss_variance)
    print('valid error: %f' % err)

# 保存训练完的权值
np.savez(file_weights_saved, weights_list)
print('weights saved')
