# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from theano import config

n_dot = 200
# 生成函数自变量数组
x = np.linspace(0, 20, n_dot)
# 计算正弦函数值
y = (np.sin(x) * 10.).astype(config.floatX)
# 线性函数
z = x * 0.1
# 随机噪声
n = np.random.random(n_dot) * 0.1
# print(n)
# 添加随机噪声
result = y + z + n

# 保存数组
data_save = zip(result, z)
# print(data_save)
np.savetxt('data\\sin.txt', data_save)
# 提取数组
# data_load = np.loadtxt('data\\sin.txt')
# print(data_load.shape)
# print(len(data_load))

# 画图
# 纵坐标最大值
y_max = max(z) + 10
plt.figure(figsize=(8, 4))
plt.plot(x, result, label="$sin(x)$", color="blue", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("PyPlot First Example")
plt.ylim(-y_max, y_max)
plt.legend()
plt.show()