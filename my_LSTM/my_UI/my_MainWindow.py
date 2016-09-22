# coding:utf-8

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# from matplotlib.ticker import  MultipleLocator
from PyQt4 import QtGui
from PyQt4 import QtCore
# import numpy as np
# import sys
import time
# import cv2.cv as cv

from my_thread import TrainThread
from my_chart import Chart

# _fromUtf8 = QtCore.QString.fromUtf8


class MenuButton(QtGui.QAction):
    """
    重写菜单栏的项目。
    用来实现，在被触发时，发射带参数的信号，而且这个参数是和该项目绑定的。
    """
    def __init__(self, *args):
        super(MenuButton, self).__init__(*args)
        self.name = None
        self.index = 0

    # 这个 index 和 weight_list 的顺序相对应，用于调取相应的权值
    def set_index(self, index):
        self.index = index

    # 发射带有权值序号的信号
    def emit_f(self):
        self.emit(QtCore.SIGNAL('emitWeightIndex(int)'), self.index)


class MainWindow(QtGui.QMainWindow):
    """
    主窗口
    """
    def __init__(self, model, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setFixedSize(500, 300)
        self.setWindowTitle('Model')
        #
        # self.user = QtGui.QLineEdit(self)
        # self.user.setGeometry(QtCore.QRect(130, 80, 250, 30))
        #
        self.btn = QtGui.QPushButton('Train', self)
        self.btn.setGeometry(QtCore.QRect(215, 190, 80, 26))
        self.connect(self.btn, QtCore.SIGNAL('clicked()'), self.train_model)

        # 创建模型
        # self.model = MyRNNModel()
        self.model = model()

        # 初始化菜单栏
        self.init_menu_bar(self.model.weights_list)

        self.init_paras_labels()

        # 初始化状态栏
        self.statusBar()

        # self.setGeometry(300, 300, 300, 200)

        self.charts = []

        # 新建线程对象
        self.bwThread = TrainThread(self.model)
        # 连接子进程的信号和槽函数， 发射信号时所调用的函数
        self.bwThread.weights_updated_signal.connect(self.show_weight_in_charts)

        # self.show()

    # 训练模型
    @QtCore.pyqtSlot()
    def train_model(self):
        # 把按钮禁用掉
        self.btn.setDisabled(True)
        # 开始执行 run() 函数里的内容
        self.bwThread.start()

    # 生成图表，并显示相应的权值
    @QtCore.pyqtSlot()
    def show_chart(self, weight_index):
        # print(self.model.weights_list[weight_index])
        # weight_shape = self.model.weights_list[weight_index].get_value().shape
        # print(weight_shape)
        chart = Chart(self.model.weights_list[weight_index], weight_index)
        self.charts.append(chart)
        chart.show_weight(self.model.weights_list[weight_index])
        chart.show()

    # 处理callback传过来的权值
    def show_weight_in_charts(self):
        # 关闭train的callback使能
        self.model.set_callback_enable(False)
        t = time.time()
        # self.canvas.fig.clear()

        for chart in self.charts:
            chart.show_weight(self.model.weights_list[chart.weight_index])

        print('show chart use time: %f' % (time.time() - t))
        # 打开train的callback使能
        self.model.set_callback_enable(True)
        # 恢复按钮
        # self.button.setDisabled(False)

    # 初始化菜单栏
    def init_menu_bar(self, weight_list):
        menu_bar = self.menuBar()
        string_file = 'File'
        fileMenu = menu_bar.addMenu(string_file)

        exitAction = QtGui.QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        # exitAction.triggered.connect(QtGui.qApp.quit)
        self.connect(exitAction, QtCore.SIGNAL('triggered()'), QtGui.qApp, QtCore.SLOT('quit()'))
        fileMenu.addAction(exitAction)

        weightMenu = menu_bar.addMenu('Weights')
        i = 0
        for weight in weight_list:
            # name = copy.deepcopy(weight.name)
            # self.names.append(name)
            # weight_button = QtGui.QAction('&%s' % name, self)
            weight_button = MenuButton('&%s' % weight.name, self)
            weight_button.set_index(i)
            # 此处用了两个信号，是为了解决自带信号 triggered() 不能带参数的问题
            self.connect(weight_button, QtCore.SIGNAL('triggered()'), weight_button.emit_f)
            self.connect(weight_button, QtCore.SIGNAL('emitWeightIndex(int)'), self.show_chart)
            weightMenu.addAction(weight_button)
            i += 1

    def init_paras_labels(self):
        for key, value in self.model.local_paras.items():
            print(key, value)
