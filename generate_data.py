# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

# 生成函数自变量数组
x = np.linspace(0, 10, 101)
# 计算正弦函数值
y = np.sin(x)
# 随机噪声
z = np.random.random(101) / 10
print(z)
# 添加随机噪声
y = y + z

# 保存数组
np.savetxt('data\\sin.txt', z)
# 提取数组
a = np.loadtxt('data\\sin.txt')
print(a)

# 画图
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="$sin(x)$", color="blue", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("PyPlot First Example")
plt.ylim(-1.2, 1.2)
plt.legend()
plt.show()