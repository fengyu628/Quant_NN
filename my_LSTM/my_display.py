# coding:utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
import  sys

import time

from my_LSTM import MyRNNModel

class MplCanvas(FigureCanvas):
    """
    Creates a canvas on which to draw our widgets
    """
    def __init__(self):
        # self.fig = Figure()
        self.fig = plt.gcf()
        # self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        # FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)

class TrainThread(QtCore.QThread):
    # 声明一个信号，同时返回一个list，同理什么都能返回啦
    weights_updated_signal = QtCore.pyqtSignal(list)

    # 构造函数里增加形参
    def __init__(self, model, parent=None):
        super(TrainThread, self).__init__(parent)
        # 储存参数
        self.model = model

    def run(self):
        self.model.set_callback_weight_updated(self.weights_updated_signal.emit)
        self.model.train()

        # self.finishSignal.emit(['hello,', 'world', '!'])


class Example1(QtGui.QWidget):
    def __init__(self,parent=None):
        super(Example1,self).__init__(parent)

        self.model = MyRNNModel()
        # 返回当前的figure
        # figure = plt.gcf()
        self.canvas = MplCanvas()
        n = 10
        x = np.asarray([i for i in range(n)] * n) * 0.1
        y = np.asarray([[i] * n for i in range(n)]).flatten() * 0.1
        # color = range(n ** 2)
        color = np.random.random(n ** 2)
        plt.scatter(x, y, c=color, s=1000, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()
        self.button = QtGui.QPushButton('train', self)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.connect(self.button, QtCore.SIGNAL('clicked()'), self, QtCore.SLOT("train()"))

    @QtCore.pyqtSlot()
    def train(self):
        # self.model.train()
        # 把按钮禁用掉
        self.button.setDisabled(True)
        # 新建对象，传入参数
        self.bwThread = TrainThread(self.model)
        # 连接子进程的信号和槽函数
        self.bwThread.weights_updated_signal.connect(self.show_weight)
        # 开始执行 run() 函数里的内容
        self.bwThread.start()

    def show_weight(self, list):
        print('get weight!')
        weight_show = list[4].get_value()
        print(weight_show)
        n = weight_show.shape[0]
        x =  np.asarray([i for i in range(n)] * n) * 0.1
        y = np.asarray([[i]*n for i in range(n)]).flatten() * 0.1
        self.canvas.fig.clear()
        plt.scatter(x, y, c=weight_show, s=1000, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()
        # 使用传回的返回值
        # for word in ls:
        #     print(word)
        # 恢复按钮
        # self.button.setDisabled(False)

    def change(self):
        print('change')
        self.canvas.fig.clear()
        n = 10
        x = np.asarray([i for i in range(n)] * n) * 0.1
        y = np.asarray([[i] * n for i in range(n)]).flatten() * 0.1
        # color = range(n ** 2)
        color = np.random.random(n ** 2)
        plt.scatter(x, y, c=color, s=1000, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ui = Example1()
    ui.show()
    app.exec_()
    print('end')