# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

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

    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    z = np.cos(x ** 2)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="$sin(x)$", color="red", linewidth=2)
    plt.plot(x, z, "b--", label="$cos(x^2)$")
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("PyPlot First Example")
    plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.show()