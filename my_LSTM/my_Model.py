# coding:utf-8

# import numpy as np
# import theano
# import theano.tensor as tensor
# from theano import config
import time

from my_layer import *
from my_optimizer import *
from my_loss import loss_variance


class MyRNNModel(object):
    """
    递归网络模型类
    """
    def __init__(self,
                 layer_type=Layer_LSTM,
                 input_dim=2,
                 inner_units=20,
                 optimizer=optimizer_sgd,
                 loss=loss_variance,
                 learning_rate=0.001,
                 epoch=100
                 ):
        super(MyRNNModel, self).__init__()
        self.input_dim = input_dim
        self.inner_units = inner_units
        self.layer_type = layer_type
        self.layer = None
        self.weights_list = []
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.epoch = epoch
        # self.local_paras = locals()

        self.callback = None
        self.callback_enable = True

    # 生成模型实体，以及权值
    def build_layer(self):
        self.layer = self.layer_type(self.input_dim, self.inner_units)
        self.weights_list = self.layer.get_weight_list()

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

    # 设置回调函数，在权值更新的调用
    def set_callback_when_weight_updated(self, callback):
        self.callback = callback

    # 设置回调函数使能
    def set_callback_enable(self, bool_value):
        self.callback_enable = bool_value

    # 训练模型
    def train(self):

        file_train_data = '..\\data\\sin.txt'
        file_valid_data = '..\\data\\sin2.txt'
        file_weights_saved = '..\\data\\LSTM_weights'

        data_train = np.loadtxt(file_train_data).astype(config.floatX)
        data_valid = np.loadtxt(file_valid_data).astype(config.floatX)

        # 生成layer输出函数
        function_layer_output = self.make_function_layer_output(self.layer)
        # 生成损失计算函数和权值更新函数
        function_compute_loss, function_update_weights = self.optimizer(self.layer, self.loss, self.weights_list)

        # 生成测试数据
        x_length = 20
        x_train_list, y_train_list, train_list_length = self.slice_data(data_train, x_length)
        x_valid_lise, y_valid_list, valid_list_length = self.slice_data(data_valid, x_length)

        # 开始训练
        for epoch_index in range(self.epoch):
            print('========== epoch: %d ==========' % epoch_index)
            t = time.time()
            for train_index in range(train_list_length):
                print('train_index: %d' % train_index)

                x_train = x_train_list[train_index]
                y_train = y_train_list[train_index]

                # 计算目标函数
                loss = function_compute_loss(x_train, y_train)
                if train_index % 20 == 0:
                    print('loss: %f' % loss)
                    # if self.callback:
                    #     self.callback(self.weights_list)

                # 更新权值
                function_update_weights(self.learning_rate)
                if self.callback and self.callback_enable is True:
                    print('show....................')
                    self.callback(self.weights_list)

            # 计算验证误差
            err = self.error_valid(x_valid_lise, y_valid_list, valid_list_length, function_layer_output, loss_variance)
            print('valid error: %f' % err)
            print('time use: %f' % (time.time() - t))

        # 保存训练完的权值
        np.savez(file_weights_saved, self.weights_list)
        print('weights saved')

# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':

    # model = MyRNNModel(layer_type=LSTM, input_dim=2, inner_units=20)
    # model.train(optimizer=sgd, loss=loss_variance, learning_rate=0.001, epoch=100)
    model = MyRNNModel()
    model.train()
