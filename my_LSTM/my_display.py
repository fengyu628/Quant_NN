# coding:utf-8

# import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
import sys
import time

from my_Model import MyRNNModel

_fromUtf8 = QtCore.QString.fromUtf8


class TrainThread(QtCore.QThread):
    """
    训练的模型的线程，因为训练时会阻塞主程序，故新起一个线程。
    """
    # 声明一个信号，同时返回一个list，同理什么都能返回啦
    weights_updated_signal = QtCore.pyqtSignal(list)

    def __init__(self, model, parent=None):
        super(TrainThread, self).__init__(parent)
        # 添加模型的引用
        self.model = model

    # 实例的start()调用
    def run(self):
        # 设置模型训练时的回调函数。回调函数为发射信号，参数是list型式（也就是权值列表）
        self.model.set_callback_when_weight_updated(self.weights_updated_signal.emit)
        self.model.train()


class MplCanvas(FigureCanvas):
    """
    创建自己的画布
    """
    def __init__(self, weight_shape):
        # 计算画布的尺寸
        size = (float(weight_shape[1])/4 + 2, float(weight_shape[0])/4 + 2)
        print(size)
        self.fig = Figure(figsize=size)
        # self.fig = plt.gcf()
        # self.axes = plt.gca()
        self.p1 = self.fig.add_subplot(111)
        self.p1.spines['right'].set_color('none')
        self.p1.spines['top'].set_color('none')
        self.p1.spines['bottom'].set_color('none')
        self.p1.spines['left'].set_color('none')

        # self.subplot = plt.subplot()
        # self.ax = self.fig.add_subplot(111)
        # FigureCanvas.__init__(self, self.fig)
        super(MplCanvas, self).__init__(self.fig)
        # super(MplCanvas, self).__init__(self.axes)
        # FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)


class Chart(QtGui.QWidget):
    """
    单独的窗口，用来显示一个权值矩阵
    """
    def __init__(self, weight_index, weight_shape, parent=None):
        super(Chart, self).__init__(parent)

        # 返回当前的figure
        self.canvas = MplCanvas(weight_shape)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self.weight_index = weight_index
        self.weight_shape = weight_shape

    def show_weight(self, weight_t):
        weight = weight_t.get_value()
        # 计算权值矩阵的尺寸
        y_length = weight.shape[0]
        x_length = weight.shape[1]
        # x == [1,2,3,4,...,1,2,3,4...]
        x = np.asarray([i for i in range(x_length)] * y_length) * 0.1
        # y == [1,1,1,1,...,4,4,4,4,...]
        y = np.asarray([[i] * x_length for i in range(y_length)]).flatten() * 0.1
        self.canvas.p1.clear()
        self.canvas.p1.set_xticks([])
        self.canvas.p1.set_yticks([])
        self.canvas.p1.set_title(weight_t.name, fontsize=14)
        self.canvas.p1.set_xlabel("min:%f   max:%f" % (weight.min(), weight.max()), fontsize=14)
        self.canvas.p1.scatter(x, y, c=weight, s=200, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()


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
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setFixedSize(500, 300)
        # self.setWindowTitle('Python GUI')
        #
        # self.user = QtGui.QLineEdit(self)
        # self.user.setGeometry(QtCore.QRect(130, 80, 250, 30))
        #
        self.btn = QtGui.QPushButton(_fromUtf8('训练'), self)
        self.btn.setGeometry(QtCore.QRect(215, 190, 80, 26))
        # self.btn.clicked.connect(self._train_model)
        self.connect(self.btn, QtCore.SIGNAL('clicked()'), self.train_model)

        # 创建模型
        self.model = MyRNNModel()

        # 初始化菜单栏
        self.init_menu_bar(self.model.weights_list)
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
        print(self.model.weights_list[weight_index])
        weight_shape = self.model.weights_list[weight_index].get_value().shape
        print(weight_shape)
        chart = Chart(weight_index, weight_shape)
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
        fileMenu = menu_bar.addMenu('&File')

        exitAction = QtGui.QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        # exitAction.triggered.connect(QtGui.qApp.quit)
        self.connect(exitAction, QtCore.SIGNAL('triggered()'), QtGui.qApp, QtCore.SLOT('quit()'))
        fileMenu.addAction(exitAction)

        weightMenu = menu_bar.addMenu('&Weights')
        i = 0
        for weight in weight_list:
            # name = copy.deepcopy(weight.name)
            # self.names.append(name)
            # weight_button = QtGui.QAction('&%s' % name, self)
            weight_button = MenuButton('&%s' % weight.name, self)
            weight_button.set_index(i)
            # weight_button.triggered.connect(lambda: self.show_weight())
            self.connect(weight_button, QtCore.SIGNAL('triggered()'), weight_button.emit_f)
            self.connect(weight_button, QtCore.SIGNAL('emitWeightIndex(int)'), self.show_chart)
            weightMenu.addAction(weight_button)
            i += 1

# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
