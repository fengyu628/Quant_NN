# coding:utf-8

import theano.tensor as tensor

def _step(m_, x_, h_, c_):
    # lstm_U.shape (128L, 512L)
    # h_:block上一次的输出
    preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
    # x_:本次的外部输入
    preact += x_
    # 最后得到的preact为输入与block上一次的输出，经过权值运算后的结果：W*x + U*y(t-1) + b

    # 1.隐含层的个数 dim_proj=128  #既是词向量的维度，又是LSTM中Hideden Units的个数
    # 输入们的输出
    i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
    # 忘记门的输出
    f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
    # 输出们的输出
    o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
    # block的输入
    c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

    # c_：上一个cell的状态， 得到的c为cell的输出
    c = f * c_ + i * c
    # m_: mask
    # c = m_[:, None] * c + (1. - m_)[:, None] * c_

    # h:block的输出
    h = o * tensor.tanh(c)
    # h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h, c