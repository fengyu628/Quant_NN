# coding:utf-8

import theano
import theano.tensor as tensor
# import numpy as np


# 随机梯度下降
# def optimizer_sgd(layer, loss_function, weights_list):
#
#     x_symbol = tensor.matrix(name='scan_input')
#     y_out_symbol = layer(x_symbol)
#
#     # 目标值
#     y_target_symbol = tensor.scalar(name='y')
#     loss_symbol = loss_function(y_out_symbol, y_target_symbol)
#
#     # 计算梯度
#     grads = tensor.grad(loss_symbol, wrt=weights_list)
#
#     grads_shared = [theano.shared(p.get_value() * 0., name=p.name) for p in weights_list]
#     grads_update = [(gs, g) for gs, g in zip(grads_shared, grads)]
#     # 制作损失函数
#     print('make loss function')
#     f_loss = theano.function([x_symbol, y_target_symbol], loss_symbol,
#                              updates=grads_update,
#                              name='sgd_f_grad_shared')
#
#     # 学习率
#     lr_symbol = tensor.scalar(name='learning rate')
#     weights_update = [(p, p - lr_symbol*g) for p, g in zip(weights_list, grads_shared)]
#     # 制作权值更新函数
#     print('make update function')
#     f_update = theano.function([lr_symbol], grads_shared,
#                                updates=weights_update,
#                                name='sgd_f_update')
#
#     return f_loss, f_update


# def optimizer_sgd_batch(layer, loss_function, weights_list, grads_list):
#
#     x_symbol_list = tensor.tensor3(dtype=theano.config.floatX)
#     # 目标值
#     y_target_symbol_list = tensor.vector(dtype=theano.config.floatX)
#
#     # 计算所有输出
#     y_out_list, scan_update = theano.scan(lambda x: layer(x),
#                                           sequences=x_symbol_list)
#     # 计算损失函数（以数组的型式，一次性计算）
#     loss_total = loss_function(y_out_list, y_target_symbol_list)
#     # 计算梯度
#     grads = tensor.grad(loss_total, weights_list)
#     # 学习率
#     lr_symbol = tensor.scalar(name='learning rate')
#     # 更新权值
#     weights_update = [(p, p - lr_symbol*g) for p, g in zip(weights_list, grads)]
#     # 取得梯度数据，用于显示或查看
#     grads_update = [(gl, g) for gl, g in zip(grads_list, grads)]
#     weights_update += grads_update
#     # 制作损失函数
#     print('make loss function')
#     f_loss = theano.function([x_symbol_list, y_target_symbol_list, lr_symbol],
#                              outputs=loss_total,
#                              updates=weights_update,
#                              name='f_sgd_mini_batch')
#     return f_loss


class Optimizer_SGD(object):
    def __init__(self, lr=0.1):
        super(Optimizer_SGD, self).__init__()
        self.lr = lr

    def make_updates(self, weights_list, grads_list, loss):
        # 计算梯度
        grads = tensor.grad(loss, weights_list)
        # 更新权值
        weights_update = [(p, p - self.lr*g) for p, g in zip(weights_list, grads)]
        # 取得梯度数据，用于显示或查看
        grads_update = [(gl, g) for gl, g in zip(grads_list, grads)]
        weights_update += grads_update
        return weights_update


class Optimizer_Other(object):
    def __init__(self):
        pass
