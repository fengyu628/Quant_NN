# coding:utf-8

import numpy as np
import theano
import theano.tensor as tensor
from theano import config
import time

from my_layer import LSTM
from my_optimizer import sgd
from my_loss import loss_variance


class MyRNNModel(object):

    def __init__(self, layer_type=LSTM, input_dim=2, inner_units=10):
        super(MyRNNModel, self).__init__()
        # self.layer_type = layer_type
        self.input_dim = input_dim
        self.inner_units = inner_units
        self.layer = layer_type(input_dim, inner_units)
        self.weights_list = self.layer.get_weight_list()

# lstm_layer, weights_list = build_rnn_model(LSTM, input_dim, inner_units)

# lstm_layer = LSTM(input_dim, inner_units)
# weights_list = lstm_layer.get_weight_list()

# import matplotlib.pyplot as plt
# weight_show = weights_list[4].get_value()
# print(weight_show)
# n = inner_units
# x =  np.asarray([i for i in range(n)] * n) * 0.1
# y = np.asarray([[i]*n for i in range(n)]).flatten() * 0.1
# plt.figure(figsize=(9, 6))
# plt.scatter(x, y, c=weight_show, s=1000, alpha=0.4, marker='s', linewidths=1)
# plt.show()
# exit(5)

    # 制作layer输出函数
    @staticmethod
    def make_function_layer_output(layer):
        x_symbol = tensor.matrix(name='scan_input')
        print('make output function')
        return theano.function([x_symbol], layer(x_symbol), name='f_out')

    # 把数据切成用于训练的小段
    @staticmethod
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
    @staticmethod
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

    def set_callback_weight_updated(self, callback):
        self.callback = callback

    # 训练模型
    def train(self, optimizer=sgd, loss=loss_variance, learning_rate=0.001, epoch=10):

        file_train_data = '..\\data\\sin.txt'
        file_valid_data = '..\\data\\sin2.txt'
        file_weights_saved = '..\\data\\LSTM_weights'

        data_train = np.loadtxt(file_train_data).astype(config.floatX)
        data_valid = np.loadtxt(file_valid_data).astype(config.floatX)

        # 生成layer输出函数
        function_layer_output = self.make_function_layer_output(self.layer)
        # 生成损失计算函数和权值更新函数
        function_compute_loss, function_update_weights = optimizer(self.layer, loss, self.weights_list)

        # 生成测试数据
        x_length = 20
        x_train_list, y_train_list, train_list_length = self.slice_data(data_train, x_length)
        x_valid_lise, y_valid_list, valid_list_length = self.slice_data(data_valid, x_length)

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
            err = self.error_valid(x_valid_lise, y_valid_list, valid_list_length, function_layer_output, loss_variance)
            print('valid error: %f' % err)
            print('time use: %f' % (time.time() - t))
            if self.callback:
                self.callback(self.weights_list)

        # 保存训练完的权值
        np.savez(file_weights_saved, self.weights_list)
        print('weights saved')




# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':



    # n_steps = len(data_train)
    # print('n_steps: %d' % n_steps)

    # model = MyRNNModel(layer_type=LSTM, input_dim=2, inner_units=20)
    # model.train(optimizer=sgd, loss=loss_variance, learning_rate=0.001, epoch=100)
    model = MyRNNModel()
    model.train()
