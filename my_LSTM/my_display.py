# coding:utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
import sys
import time

from my_LSTM import MyRNNModel


class TrainThread(QtCore.QThread):
    """
    训练的模型的线程，因为训练时会阻塞主程序，故新起一个线程。
    """
    # 声明一个信号，同时返回一个list，同理什么都能返回啦
    weights_updated_signal = QtCore.pyqtSignal(list)

    def __init__(self, model, parent=None):
        super(TrainThread, self).__init__(parent)
        # 储存参数
        self.model = model

    # 实例的start()调用
    def run(self):
        self.model.set_callback_weight_updated(self.weights_updated_signal.emit)
        self.model.train()


class MplCanvas(FigureCanvas):
    """
    创建自己的画布
    """
    def __init__(self):
        # self.fig = Figure()
        self.fig = plt.gcf()
        # self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        # FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)


class MainWindow(QtGui.QWidget):
    """
    主界面类
    """
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # 返回当前的figure
        self.canvas = MplCanvas()

        self.button = QtGui.QPushButton('train', self)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.connect(self.button, QtCore.SIGNAL('clicked()'), self, QtCore.SLOT("train_model()"))

        # 创建模型
        self.model = MyRNNModel()
        # 显示权值的初始值
        self.show_weight(self.model.weights_list)

        # 新建对象，传入参数
        self.bwThread = TrainThread(self.model)
        # 连接子进程的信号和槽函数
        self.bwThread.weights_updated_signal.connect(self.show_weight)

    @QtCore.pyqtSlot()
    def train_model(self):
        # 把按钮禁用掉
        self.button.setDisabled(True)
        # 开始执行 run() 函数里的内容
        self.bwThread.start()

    # 处理callback传过来的权值
    def show_weight(self, weight_list):
        t = time.time()
        weight_show = weight_list[4].get_value()
        # 计算权值矩阵的尺寸
        y_length = weight_show.shape[0]
        x_length = weight_show.shape[1]
        # x == [1,2,3,4,...,1,2,3,4...]
        x = np.asarray([i for i in range(x_length)] * y_length) * 0.1
        # y == [1,1,1,1,...,4,4,4,4,...]
        y = np.asarray([[i]*x_length for i in range(y_length)]).flatten() * 0.1
        self.canvas.fig.clear()
        plt.scatter(x, y, c=weight_show, s=1000, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()
        print('show use time: %f' % (time.time() - t))
        self.model.set_callback_interval(time.time() - t)

        # 恢复按钮
        # self.button.setDisabled(False)

# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    app.exec_()
    print('end')
