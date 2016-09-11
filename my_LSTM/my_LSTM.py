# coding:utf-8

import numpy as np
import theano
import theano.tensor as tensor
from theano import config
import time

from my_layer import LSTM
from my_optimizer import sgd
from my_loss import loss_variance


input_dim = 2
inner_units = 200
learning_rate = 0.001
epoch = 100

file_train_data = 'data\\sin.txt'
file_valid_data = 'data\\sin2.txt'
file_weights_saved = 'data\\LSTM_weights'

data_train = np.loadtxt(file_train_data).astype(config.floatX)
data_valid = np.loadtxt(file_valid_data).astype(config.floatX)

n_steps = len(data_train)
print('n_steps: %d' % n_steps)


# ======================================================================================================================
# ======================================================================================================================


lstm_layer = LSTM(input_dim, inner_units)
weights_list = lstm_layer.get_weight_list()


# 制作layer输出函数
def make_function_layer_output(layer):
    x_symbol = tensor.matrix(name='scan_input')
    print('make output function')
    return theano.function([x_symbol], layer(x_symbol), name='f_out')


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
function_compute_loss, function_update_weights = sgd(lstm_layer, loss_variance, weights_list)

# 生成测试数据
x_length = 20
x_train_list, y_train_list, train_list_length = slice_data(data_train, x_length)
x_valid_lise, y_valid_list, valid_list_length = slice_data(data_valid, x_length)

# ======================================================================================================================
# ======================================================================================================================

# 开始训练
for epoch_index in range(epoch):
    print('========== epoch: %d ==========' % epoch_index)
    t = time.time()
    for train_index in range(train_list_length):
        # print('train_index: ', train_index)

        x_train = x_train_list[train_index]
        y_train = y_train_list[train_index]

        # 计算目标函数
        loss = function_compute_loss(x_train, y_train)
        if train_index % 20 == 0:
            print('loss: %f' % loss)

        # 更新权值
        function_update_weights(learning_rate)

    # 计算验证误差
    # err = error_valid(x_valid_lise, y_valid_list, valid_list_length, function_layer_output, loss_variance)
    # print('valid error: %f' % err)
    print('time use: %f' % (time.time() - t))

# 保存训练完的权值
np.savez(file_weights_saved, weights_list)
print('weights saved')
