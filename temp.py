# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tensor
from theano import config

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as figureCanvas
from matplotlib.figure import Figure
import sys

from my_LSTM import my_loss

#-*- coding:utf-8 -*-
#######pyqt  文件载入对话框，文件保存对话框，打开文件夹对话框
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys

b = [5]
a = range(5)
# print(a)
# for i in a:
#     b.append(i)
print(b)
for e in b:
    print(e)

# class MyWindow(QDialog,QWidget):
#     def __init__(self,parent = None):
#         super(MyWindow,self).__init__(parent)
#         self.resize(400,150)
#         self.mainlayout = QGridLayout(self)
#         self.loadFileButton = QPushButton()
#         self.loadFileButton.setText(u"载入文件")
#         self.mainlayout.addWidget(self.loadFileButton,0,0,1,1)
#         self.loadFileLineEdit = QLineEdit()
#         self.mainlayout.addWidget(self.loadFileLineEdit,0,1,1,4)
#         self.loadFileButton.clicked.connect(self.loadFile)
#
#         self.saveFileButton = QPushButton()
#         self.saveFileButton.setText(u"保存文件")
#         self.saveFileLineEdit = QLineEdit()
#         self.mainlayout.addWidget(self.saveFileButton,1,0,1,1)
#         self.mainlayout.addWidget(self.saveFileLineEdit,1,1,1,4)
#         self.saveFileButton.clicked.connect(self.saveFile)
#
#         self.openFileDirButton = QPushButton()
#         self.openFileDirButton.setText(u"打开文件目录")
#         self.mainlayout.addWidget(self.openFileDirButton,2,0,1,1)
#         self.openFileDirButton.clicked.connect(self.openFileDirectory)
#
#     def loadFile(self):########载入file
#         file_name = QFileDialog.getOpenFileName(self,"open file dialog","C:\Users\Administrator\Desktop","Txt files(*.txt)")
#         ##"open file Dialog "文件对话框的标题，第二个是打开的默认路径，第三个是文件类型
#         self.loadFileLineEdit.setText(file_name)
#
#     def saveFile(self):
#         file_path =  QFileDialog.getSaveFileName(self,'save file',"saveFile" ,"xj3dp files (*.xj3dp);;all files(*.*)") ####
#         print file_path
#
#     def openFileDirectory(self):
#         import os
#         os.popen("explorer.exe C:\Users\Administrator\Desktop")
#
#
# app=QApplication(sys.argv)
# window=MyWindow()
# window.show()
# app.exec_()



# print(dir(my_loss))
# for i in dir(my_loss):
#     print(type(i))


# class father(object):
#     def __init__(self):
#         self.name = 'Father'
#
#     def __call__(self):
#         print('father call')
#         self.build()
#         print(self.name)
#
#     def build(self):
#         print('father build')
#
#
# class son(father):
#     def __init__(self):
#         super(son, self).__init__()
#         self.name = 'Son'
#
#     def call(self):
#         print('son call')
#
#     def build(self):
#         print('son build')
#
#
# class Example1(QWidget):
#     def __init__(self, parent=None):
#         super(Example1, self).__init__(parent)
#         # 返回当前的figure
#         figure = plt.gcf()
#         self.canvas = figureCanvas(figure)
#         x = [1, 2, 3]
#         y = [4, 5, 6]
#         plt.plot(x, y)
#         plt.title('Example')
#         plt.xlabel('x')
#         plt.ylabel('y')
#         self.canvas.draw()
#         layout = QHBoxLayout(self)
#         layout.addWidget(self.canvas)


# if __name__ == "__main__":
#
#     if __name__ == '__main__':
#         app = QApplication(sys.argv)
#         ui = Example1()
#         ui.show()
#         app.exec_()

'''
    x = np.linspace(0, 5 * np.pi, 400)
    print(x)
    exit(6)

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

    color = range(n**2, 0, -1)
    print(color)
    plt.scatter(x, y, c=color, s=1000, alpha=0.4, marker='s', linewidths=1)
    plt.show()
'''
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