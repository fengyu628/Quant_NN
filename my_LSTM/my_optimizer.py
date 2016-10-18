# coding:utf-8

import theano
import theano.tensor as tensor


# 随机梯度下降
def optimizer_sgd(layer, loss_function, weights_list):

    x_symbol = tensor.matrix(name='scan_input')
    y_out_symbol = layer(x_symbol)

    # 目标值
    y_target_symbol = tensor.scalar(name='y')
    loss_symbol = loss_function(y_out_symbol, y_target_symbol)

    # 计算梯度
    grads = tensor.grad(loss_symbol, wrt=weights_list)

    grads_shared = [theano.shared(p.get_value() * 0., name=p.name) for p in weights_list]
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
    f_update = theano.function([lr_symbol], grads_shared,
                               updates=weights_update,
                               name='sgd_f_update')

    return f_loss, f_update


def optimizer_other():
    pass
