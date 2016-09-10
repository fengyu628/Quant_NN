# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tensor
from theano import config

class father(object):
    def __init__(self):
        self.name = 'Father'

    def __call__(self):
        print('father call')
        self.build()
        print(self.name)

    def build(self):
        print('father build')


class son(father):
    def __init__(self):
        super(son, self).__init__()
        self.name = 'Son'

    def call(self):
        print('son call')

    def build(self):
        print('son build')


if __name__ == '__main__':


    a = [[i]*5 for i in range(5)]
    a = np.asarray(a).flatten()
    # print(a)

    n = 5
    a = np.random.rand(n, n)
    # print(a)
    a_f = a.flatten()
    # print(a_f)

    plt.figure(figsize=(9, 6))
    n = 10
    # rand 均匀分布和 randn高斯分布
    # x = np.random.randn(1, n)
    # y = np.random.randn(1, n)
    # print(x)
    # print(y)
    # x = np.arange(25)
    x =  np.asarray([i for i in range(n)] * n) * 0.1
    y = np.asarray([[i]*n for i in range(n)]).flatten() * 0.1
    color = range(n**2)
    # color.shape = (5, 5)
    # print(color)
    # y = np.arange(25)
    # print(x)
    # T = np.arctan2(x, y)
    # print(T)
    plt.scatter(x, y, c=color, s=1000, alpha=0.4, marker='s', linewidths=1)
    # T:散点的颜色
    # s：散点的大小
    # alpha:是透明程度
    plt.show()

    color = color.reverse()
    print(color)

    # rand = np.random.random()
    # print(rand)
'''
    a = np.asarray([1,2])#.astype(config.floatX)
    b = np.asarray([[1,2,3],[4,5,6]])#.astype(config.floatX)
    # a = theano.shared(a)
    # b = theano.shared(b)
    print(a,b)
    c = tensor.dot(a, b).eval()
    print(c)
    d = tensor.nnet.sigmoid(c).eval()
    print(d)

    c = np.asarray([1, 2, 3])
    d = np.asarray([4, 5, 6])
    e = tensor.dot(c, d).eval()
    print(e)
'''
    # f = father()
    # print(f.name)
    # s = son()
    # print(s.name)
    # s()
    # l = [1, 2, 3]
    # print(l[-1])

    # a = np.linspace(0, 10, 101)
    # print(a)
    #
    # x = np.arange(0., 10., 0.1)
    # print(x)
    # sin_data = np.sin(x)
    # print(sin_data)

    # x = np.linspace(0, 10, 1000)
    # y = np.sin(x)
    # z = np.cos(x ** 2)
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(x, y, label="$sin(x)$", color="red", linewidth=2)
    # plt.plot(x, z, "b--", label="$cos(x^2)$")
    # plt.xlabel("Time(s)")
    # plt.ylabel("Volt")
    # plt.title("PyPlot First Example")
    # plt.ylim(-1.2, 1.2)
    # plt.legend()
    # plt.show()