# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

n_dot = 200
# 生成函数自变量数组
x = np.linspace(0, 20, n_dot)
# 计算正弦函数值
y = np.sin(x)
# 线性函数
z = x * 0.1
# 随机噪声
n = np.random.random(n_dot) / 10
# print(n)
# 添加随机噪声
result = y + z + n

# 保存数组
np.savetxt('data\\sin.txt', [result, z])
# 提取数组
a, b = np.loadtxt('data\\sin.txt')
print(a, b)

# 画图
# 纵坐标最大值
y_max = max(z) + 1
plt.figure(figsize=(8, 4))
plt.plot(x, result, label="$sin(x)$", color="blue", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("PyPlot First Example")
plt.ylim(-1.2, y_max)
plt.legend()
plt.show()