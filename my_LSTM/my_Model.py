# coding:utf-8

import time
import copy

from my_layer import *
from my_optimizer import *
from my_loss import *
from my_regularizers import *


class MyRNNModel(object):
    """
    递归网络模型类
    """
    def __init__(self,
                 layer_type=Layer_LSTM,
                 input_dim=8,
                 inner_units=20,
                 loss=loss_variance,
                 optimizer_type=Optimizer_SGD,
                 mini_batch_size=20,
                 epoch=100
                 ):
        super(MyRNNModel, self).__init__()
        # self.__dict__.update(locals())
        self.input_dim = input_dim
        self.inner_units = inner_units
        self.layer_type = layer_type
        self.layer = None
        self.weights_list = []
        self.loss = loss
        self.optimizer_type = optimizer_type
        self.optimizer = None
        self.mini_batch_size = mini_batch_size
        self.epoch = epoch

        self.callback = None
        self.callback_enable = True

        self.pause_train_flag = False
        self.stop_train_flag = False

        self.train_x = []
        self.train_y = []
        self.validate_x = []
        self.validate_y = []

        self.grads_list = []

        self.regularizer = WeightRegularizer()

    # 在保存模型之前需要做的操作
    def set_status_before_save(self):
        self.callback = None
        self.callback_enable = True
        self.pause_train_flag = False
        self.stop_train_flag = False
        # 不保存训练数据
        self.train_x = []
        self.train_y = []
        self.validate_x = []
        self.validate_y = []

        self.grads_list = []

    # 设置回调函数，在权值更新的调用
    def set_callback_when_weight_updated(self, callback):
        self.callback = callback

    # 设置回调函数使能
    def set_callback_enable(self, bool_value):
        self.callback_enable = bool_value

    def pause_training(self):
        self.pause_train_flag = True

    def resume_training(self):
        self.pause_train_flag = False

    def stop_training(self):
        self.stop_train_flag = True

    # ----------------- 生成模型实体，以及权值 ------------------
    def build_layer(self):
        # 实例化模型
        self.layer = self.layer_type(self.input_dim, self.inner_units)
        self.weights_list = self.layer.get_weight_list()
        # 初始化 grads_list，此处只是取 weights_list 的型式
        self.grads_list = copy.deepcopy(self.weights_list)

    # ------------- 制作损失和更新函数，最重要的步骤 -------------
    def make_function_loss_update(self):
        x_symbol_list = tensor.tensor3(dtype=theano.config.floatX)
        # 目标值
        y_target_symbol_list = tensor.vector(dtype=theano.config.floatX)

        # 计算所有输出
        y_out_list, scan_update = theano.scan(lambda x: self.layer(x),
                                              sequences=x_symbol_list)
        # 计算损失函数（以数组的型式，一次性计算）
        loss_no_regularizer = self.loss(y_out_list, y_target_symbol_list)
        # 添加正则项
        self.regularizer.set_param(self.weights_list)
        loss_total = self.regularizer(loss_no_regularizer)
        # 制作损失函数
        print('make loss function')
        f_loss = theano.function([x_symbol_list, y_target_symbol_list],
                                 outputs=loss_total,
                                 name='f_loss')
        # 通过优化器得到更新数据
        updates = self.optimizer.get_updates(self.weights_list, self.grads_list, loss_total, [])
        # 制作带更新权值的损失函数
        print('make loss and update function')
        f_loss_update = theano.function([x_symbol_list, y_target_symbol_list],
                                        outputs=loss_total,
                                        updates=updates,
                                        name='f_loss_update')
        return f_loss, f_loss_update

    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------- 训练模型 --------------------------------------------
    # -------------------------------------------------------------------------------------------------
    def train(self):

        self.optimizer = self.optimizer_type()

        print('*' * 50)
        print('Layer: %s' % self.layer.__class__)
        print('Input Dim: %d' % self.input_dim)
        print('Inner Units: %d' % self.inner_units)
        print('Loss: %s' % self.loss.__name__)
        print('Regularizer: %s' % self.regularizer.__class__)
        print('Optimizer: %s' % self.optimizer.__class__)
        print('Leaning Rate: %s' % self.optimizer.lr.get_value())
        print('Mini Batch Size: %d' % self.mini_batch_size)
        print('Epoch: %d' % self.epoch)
        print('*' * 50)

        x_train_list = self.train_x
        y_train_list = self.train_y
        x_valid_list = self.validate_x
        y_valid_list = self.validate_y

        if len(x_train_list) != len(y_train_list):
            print('train: x length is not equal to y length')
            return
        if len(x_valid_list) != len(y_valid_list):
            print('validate: x length is not equal to y length')
            return

        # 生成损失计算函数和权值更新函数
        function_loss, function_loss_and_update = self.make_function_loss_update()

        # 用于发送实时的训练结果
        temp_loss_list = []
        temp_error_list = []

        # ************************* 开始训练 ***********************
        for epoch_index in range(self.epoch):
            print('========== epoch: %d ==========' % epoch_index)
            print('lr: %f' % self.optimizer.lr.eval())
            t = time.time()
            # 丢掉不足一个mini_batch的数据
            for mini_batch_index in range(int(len(y_train_list)/int(self.mini_batch_size))):
                # 根据标志位判断是否停止训练
                if self.stop_train_flag is True:
                    print('Stop train!')
                    return
                while True:
                    # 根据标志位判断是否暂停训练
                    if self.pause_train_flag is False:
                        break
                    # 根据标志位判断是否停止训练
                    if self.stop_train_flag is True:
                        print('Stop train!')
                        return
                    time.sleep(0.1)
                # 创建训练数据的 miniBatch
                x_train_mini_batch = np.asarray(x_train_list[int(mini_batch_index * self.mini_batch_size):
                                                             int((mini_batch_index+1) * self.mini_batch_size)])
                y_train_mini_batch = np.asarray(y_train_list[int(mini_batch_index * self.mini_batch_size):
                                                             int((mini_batch_index+1) * self.mini_batch_size)])
                assert len(x_train_mini_batch) == len(y_train_mini_batch)
                # 计算目标函数，并更新权值
                loss = function_loss_and_update(x_train_mini_batch, y_train_mini_batch)

                if mini_batch_index % 10 == 0:
                    print('mini batch index: %d' % mini_batch_index)
                    print('loss: %f' % loss)

                # if train_index % 10 == 0:
                #     print('*********** grads ***********')
                #     print(id(self.weights_list))
                #     print(id(self.layer.weights_list))
                #     print(id(self.grads_list))
                #     for grad in grads:
                #         print(grad)
                # 发送loss的列表,此处如果不加‘float’，temp_loss_list会变成array(XXXX)
                temp_loss_list.append(float(loss))
                # 调用回调函数
                if self.callback and self.callback_enable is True:
                    callback_dict = {'weights_list': self.weights_list, 'temp_loss_list': temp_loss_list}
                    # print('call back ....................')
                    # print(temp_loss_list)
                    self.callback(callback_dict)
                    # 发送完后清空
                    temp_loss_list = []

            # 计算验证误差
            print('computing error ...')
            err = function_loss(x_valid_list, y_valid_list)
            print('valid error: %f' % err)
            print('time use: %f' % (time.time() - t))
            temp_error_list.append(float(err))
            # 调用回调函数
            if self.callback and self.callback_enable is True:
                callback_dict = {'temp_error_list': temp_error_list}
                # print('call back ....................')
                self.callback(callback_dict)
                # 发送完后清空
                temp_error_list = []

            # 每个epoch计算一次学习率更新
            if hasattr(self.optimizer, 'update_lr'):
                print('update lr ***************************************************')
                self.optimizer.update_lr()

        if self.callback and self.callback_enable is True:
            callback_dict = {'train_end': True}
            # print('call back ....................')
            self.callback(callback_dict)

# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':

    # 把数据切成用于训练的小段
    def slice_data(data_to_slice, slice_length):
        list_length = 0
        x_list = []
        y_list = []
        for i in range(len(data_to_slice) - slice_length - 6):
            x_list.append(data_to_slice[i:i + slice_length, ])
            y_list.append(data_to_slice[i + slice_length + 5, 0])
            list_length += 1
        return x_list, y_list

    file_train_data = '..\\data\\sin.txt'
    file_valid_data = '..\\data\\sin2.txt'
    # file_weights_saved = '..\\data\\LSTM_weights'

    data_train = np.loadtxt(file_train_data).astype(config.floatX)
    data_valid = np.loadtxt(file_valid_data).astype(config.floatX)

    # 生成测试数据
    x_length = 20
    x_train_array, y_train_array = slice_data(data_train, x_length)
    x_valid_array, y_valid_array = slice_data(data_valid, x_length)

    model = MyRNNModel(layer_type=Layer_LSTM,
                       input_dim=2,
                       inner_units=20,
                       loss=loss_variance,
                       optimizer_type=Optimizer_SGD,
                       epoch=100)

    model.train_x = x_train_array
    model.train_y = y_train_array
    model.validate_x = x_valid_array
    model.validate_y = y_valid_array

    model.build_layer()
    model.train()
