# coding:utf-8

# import matplotlib.pyplot as plt
# from matplotlib.ticker import  MultipleLocator
from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
# import sys
import time
# import cv2.cv as cv

from my_thread import TrainThread, MyGeneralThread
from my_chart import Chart
from my_mplCanvas import MplCanvas

from my_LSTM import my_layer
from my_LSTM import my_loss
from my_LSTM import my_optimizer

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
        self.emit(QtCore.SIGNAL('clickMenuButtonWithWeightIndex(int)'), self.index)
        self.setDisabled(True)


class MainWindow(QtGui.QMainWindow):
    """
    主窗口
    """
    def __init__(self, model, parent=None):
        super(MainWindow, self).__init__(parent)

        # self.setFixedSize(500, 300)
        self.resize(800,600)
        self.setWindowTitle('Model')

        # 创建模型
        self.model = model()

        # 初始化菜单栏
        # 初始化 “File” 菜单
        menu_bar = self.menuBar()
        string_file = 'File'
        self.fileMenu = menu_bar.addMenu(string_file)
        self.exitAction = QtGui.QAction('Exit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        # exitAction.triggered.connect(QtGui.qApp.quit)
        self.connect(self.exitAction, QtCore.SIGNAL('triggered()'), QtGui.qApp, QtCore.SLOT('quit()'))
        self.fileMenu.addAction(self.exitAction)

        # 初始化 “Weight” 菜单
        self.weightMenu = menu_bar.addMenu('Weights')
        self.weightMenu.setDisabled(True)
        self.weightMenuItems = []

        # 初始化控件
        self.layerLabel = QtGui.QLabel('Layer Type:')
        self.layerComboBox = QtGui.QComboBox()

        self.inputDimLabel = QtGui.QLabel('Input Dim:')
        self.inputDimEdit = QtGui.QLineEdit(self)
        self.inputDimEdit.setText(str(self.model.input_dim))
        # 暂时不许更改输入维度
        self.inputDimEdit.setReadOnly(True)

        self.innerUnitsLabel = QtGui.QLabel('Inner Units:')
        self.innerUnitsEdit = QtGui.QLineEdit(self)
        self.innerUnitsEdit.setText(str(self.model.inner_units))

        self.lossFunctionLabel = QtGui.QLabel('Loss Function:')
        self.lossComboBox = QtGui.QComboBox()

        self.optimizerLabel = QtGui.QLabel('Optimizer Function:')
        self.optimizerComboBox = QtGui.QComboBox()

        self.learningRateLabel = QtGui.QLabel('learning Rate:')
        self.learningRateEdit = QtGui.QLineEdit(self)
        self.learningRateEdit.setText(str(self.model.learning_rate))

        self.epochLabel = QtGui.QLabel('Epoch:')
        self.epochEdit = QtGui.QLineEdit(self)
        self.epochEdit.setText(str(self.model.epoch))

        self.buildButton = QtGui.QPushButton('Build Model', self)
        self.buildButton.setFixedSize(100, 50)
        self.connect(self.buildButton, QtCore.SIGNAL('clicked()'), self.build_model)

        self.trainButton = QtGui.QPushButton('Train', self)
        self.trainButton.setFixedSize(100, 50)
        self.trainButton.setDisabled(True)
        self.connect(self.trainButton, QtCore.SIGNAL('clicked()'), self.train_model)

        self.pauseTrainButton = QtGui.QPushButton('Pause Train', self)
        self.pauseTrainButton.setFixedSize(100, 50)
        self.pauseTrainButton.setDisabled(True)
        self.connect(self.pauseTrainButton, QtCore.SIGNAL('clicked()'), self.pause_train)

        self.resumeTrainButton = QtGui.QPushButton('Resume Train', self)
        self.resumeTrainButton.setFixedSize(100, 50)
        self.resumeTrainButton.setDisabled(True)
        self.connect(self.resumeTrainButton, QtCore.SIGNAL('clicked()'), self.resume_train)

        self.stopTrainButton = QtGui.QPushButton('Stop Train', self)
        self.stopTrainButton.setFixedSize(100, 50)
        self.stopTrainButton.setDisabled(True)
        self.connect(self.stopTrainButton, QtCore.SIGNAL('clicked()'), self.stop_train)

        self.closeAllChartsButton = QtGui.QPushButton('Close All\nWeight Charts', self)
        self.closeAllChartsButton.setFixedSize(100, 50)
        self.connect(self.closeAllChartsButton, QtCore.SIGNAL('clicked()'), self.close_all_charts)
        
        self.lossResultLabel = QtGui.QLabel('Loss:')

        self.errorResultLabel = QtGui.QLabel('Error:')

        self.lossCanvas = MplCanvas(title='Loss')
        self.errorCanvas = MplCanvas(title='Error')

        self.init_paras_labels()

        self.connect(self.layerComboBox,
                     QtCore.SIGNAL('currentIndexChanged(int)'),
                     self.layer_combobox_changed)
        self.connect(self.lossComboBox,
                     QtCore.SIGNAL('currentIndexChanged(int)'),
                     self.loss_combobox_changed)
        self.connect(self.optimizerComboBox,
                     QtCore.SIGNAL('currentIndexChanged(int)'),
                     self.optimizer_combobox_changed)

        top_left_frame = QtGui.QFrame()
        top_left_frame.setFrameShape(QtGui.QFrame.StyledPanel)
        top_left_grid = QtGui.QGridLayout()
        top_left_frame.setLayout(top_left_grid)
        top_left_grid.addWidget(self.layerLabel, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        top_left_grid.addWidget(self.layerComboBox, 0, 1, 1, 1)
        top_left_grid.addWidget(self.inputDimLabel, 1, 0, 1, 1, QtCore.Qt.AlignRight)
        top_left_grid.addWidget(self.inputDimEdit, 1, 1, 1, 1)
        top_left_grid.addWidget(self.innerUnitsLabel, 2, 0, 1, 1, QtCore.Qt.AlignRight)
        top_left_grid.addWidget(self.innerUnitsEdit, 2, 1, 1, 1)
        top_left_grid.addWidget(self.lossFunctionLabel, 3, 0, 1, 1, QtCore.Qt.AlignRight)
        top_left_grid.addWidget(self.lossComboBox, 3, 1, 1, 1)
        top_left_grid.addWidget(self.optimizerLabel, 4, 0, 1, 1, QtCore.Qt.AlignRight)
        top_left_grid.addWidget(self.optimizerComboBox, 4, 1, 1, 1)
        top_left_grid.addWidget(self.learningRateLabel, 5, 0, 1, 1, QtCore.Qt.AlignRight)
        top_left_grid.addWidget(self.learningRateEdit, 5, 1, 1, 1)
        top_left_grid.addWidget(self.epochLabel, 6, 0, 1, 1, QtCore.Qt.AlignRight)
        top_left_grid.addWidget(self.epochEdit, 6, 1, 1, 1)

        top_right_frame = QtGui.QFrame()
        top_right_frame.setFrameShape(QtGui.QFrame.StyledPanel)
        top_right_grid = QtGui.QGridLayout()
        top_right_frame.setLayout(top_right_grid)
        top_right_grid.addWidget(self.buildButton, 0, 0, 1, 1)
        top_right_grid.addWidget(self.trainButton, 0, 1, 1, 1)
        top_right_grid.addWidget(self.pauseTrainButton, 1, 0, 1, 1)
        top_right_grid.addWidget(self.resumeTrainButton, 1, 1, 1, 1)
        top_right_grid.addWidget(self.stopTrainButton, 2, 0, 1, 1)
        top_right_grid.addWidget(self.closeAllChartsButton, 2, 1, 1, 1)

        bottom_frame = QtGui.QFrame()
        bottom_frame.setFrameShape(QtGui.QFrame.StyledPanel)
        bottom_grid = QtGui.QGridLayout()
        bottom_frame.setLayout(bottom_grid)
        bottom_grid.addWidget(self.lossResultLabel, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        bottom_grid.addWidget(self.errorResultLabel, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        bottom_grid.addWidget(self.lossCanvas, 1, 0, 1, 2, QtCore.Qt.AlignCenter)
        bottom_grid.addWidget(self.errorCanvas, 2, 0, 1, 2, QtCore.Qt.AlignCenter)

        splitter_top = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter_top.addWidget(top_left_frame)
        splitter_top.addWidget(top_right_frame)

        splitter_vertical = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter_vertical.addWidget(splitter_top)
        splitter_vertical.addWidget(bottom_frame)

        # h_box = QtGui.QHBoxLayout(self)
        # h_box.addWidget(splitter_vertical)
        # self.main_widget = QtGui.QWidget()
        # self.main_widget.setLayout(h_box)
        self.setCentralWidget(splitter_vertical)

        # 初始化状态栏
        self.statusBar()

        # self.setGeometry(300, 300, 300, 200)

        self.charts = []

        # 训练模型的线程
        self.trainThread = TrainThread(self.model)
        # 连接子进程的信号和槽函数， 发射信号时所调用的函数
        self.trainThread.weights_updated_signal.connect(self.deal_with_train_callback)

        self.drawLossCanvasThread = MyGeneralThread()
        self.drawLossCanvasThread.set_thread_function(self.lossCanvas.draw_data)
        self.drawErrorCanvasThread = MyGeneralThread()
        self.drawErrorCanvasThread.set_thread_function(self.errorCanvas.draw_data)

    # 生成模型
    @QtCore.pyqtSlot()
    def build_model(self):
        # 把按钮禁用掉
        self.buildButton.setDisabled(True)
        try:
            self.model.input_dim = int(self.inputDimEdit.text())
        except Exception as e:
            print('error input_dim')
            print(e)
            QtGui.QMessageBox.warning(self, 'warning',
                                      u'Input Dim 错误',
                                      QtGui.QMessageBox.Cancel)
            self.buildButton.setDisabled(False)
            return
        try:
            self.model.inner_units = int(self.innerUnitsEdit.text())
        except Exception as e:
            print('error inner_units')
            print(e)
            QtGui.QMessageBox.warning(self, 'warning',
                                      u'Inner Units 错误',
                                      QtGui.QMessageBox.Cancel)
            self.buildButton.setDisabled(False)
            return
        try:
            self.model.learning_rate = float(self.learningRateEdit.text())
        except Exception as e:
            print('error learning_rate')
            print(e)
            QtGui.QMessageBox.warning(self, 'warning',
                                      u'Learning Rate 错误',
                                      QtGui.QMessageBox.Cancel)
            self.buildButton.setDisabled(False)
            return
        try:
            self.model.epoch = int(self.epochEdit.text())
        except Exception as e:
            print('error epoch')
            print(e)
            QtGui.QMessageBox.warning(self, 'warning',
                                      u'Epoch 错误',
                                      QtGui.QMessageBox.Cancel)
            self.buildButton.setDisabled(False)
            return
        self.model.build_layer()
        # 生成权值菜单
        self.weightMenu.setDisabled(False)
        self.init_weight_menu(self.model.weights_list)
        # 使能训练按钮
        self.trainButton.setDisabled(False)

    # 开始训练模型
    @QtCore.pyqtSlot()
    def train_model(self):
        # 把按钮禁用掉
        self.trainButton.setDisabled(True)
        # 启动线程
        self.trainThread.start(QtCore.QThread.HighPriority)
        self.drawLossCanvasThread.start(QtCore.QThread.LowPriority)
        self.drawErrorCanvasThread.start(QtCore.QThread.LowPriority)
        # 使能停止训练按钮
        self.pauseTrainButton.setDisabled(False)
        self.stopTrainButton.setDisabled(False)

    # 暂停训练
    @QtCore.pyqtSlot()
    def pause_train(self):
        self.model.pause_training()
        self.pauseTrainButton.setDisabled(True)
        self.resumeTrainButton.setDisabled(False)

    # 重新开始训练
    @QtCore.pyqtSlot()
    def resume_train(self):
        self.model.resume_training()
        self.pauseTrainButton.setDisabled(False)
        self.resumeTrainButton.setDisabled(True)

    # 停止训练
    @QtCore.pyqtSlot()
    def stop_train(self):
        self.model.stop_training()
        self.pauseTrainButton.setDisabled(True)
        self.resumeTrainButton.setDisabled(True)
        self.stopTrainButton.setDisabled(True)

    # 生成图表，并显示相应的权值
    @QtCore.pyqtSlot()
    def show_chart(self, weight_index):
        # print(self.model.weights_list[weight_index])
        # weight_shape = self.model.weights_list[weight_index].get_value().shape
        # print(weight_shape)
        chart = Chart(self.model.weights_list[weight_index], weight_index)
        self.connect(chart, QtCore.SIGNAL('closeChartWithWeightIndex(int)'), self.close_chart_event)
        self.charts.append(chart)
        chart.show_weight(self.model.weights_list[weight_index])
        chart.show()

    # 有 Chart 被关闭的事件处理
    @QtCore.pyqtSlot()
    def close_chart_event(self, weight_index):
        print('chart weight index: %d' % weight_index)
        for chart in self.charts:
            if chart.weight_index == weight_index:
                self.charts.remove(chart)
                # 恢复权值菜单元素的使能
                for item in self.weightMenuItems:
                    if item.index == weight_index:
                        item.setDisabled(False)
        print('charts count:%d' % len(self.charts))

    # 关闭所有 Chart
    @QtCore.pyqtSlot()
    def close_all_charts(self):
        self.charts = []
        for item in self.weightMenuItems:
            item.setDisabled(False)

    # 改变模型的 layer
    @QtCore.pyqtSlot()
    def layer_combobox_changed(self):
        self.model.layer_type = getattr(my_layer, str(self.layerComboBox.currentText()))

    # 改变模型的 loss
    @QtCore.pyqtSlot()
    def loss_combobox_changed(self):
        self.model.loss = getattr(my_loss, str(self.lossComboBox.currentText()))

    # 改变模型的 optimizer
    @QtCore.pyqtSlot()
    def optimizer_combobox_changed(self):
        self.model.optimizer = getattr(my_optimizer, str(self.optimizerComboBox.currentText()))

    # 处理callback传过来的权值
    def deal_with_train_callback(self, callback_dict):
        # 关闭train的callback使能
        self.model.set_callback_enable(False)
        t = time.time()
        # self.canvas.fig.clear()
        # 更新窗口中的权值
        for chart in self.charts:
            chart.show_weight(self.model.weights_list[chart.weight_index])

        if callback_dict.has_key('temp_loss_list'):
            temp_loss_list = callback_dict['temp_loss_list']
            self.lossResultLabel.setText('Loss: %f' % temp_loss_list[-1])

            self.lossCanvas.draw_enable_flag = False
            if len(self.lossCanvas.index_list) == 0:
                for i in range(len(temp_loss_list)):
                    self.lossCanvas.index_list.append(i)
            else:
                for i in range(len(temp_loss_list)):
                    self.lossCanvas.index_list.append(self.lossCanvas.index_list[-1] + 1 + i)
            print(len(self.lossCanvas.index_list))
            for loss in temp_loss_list:
                self.lossCanvas.value_list.append(loss)
            print(len(self.lossCanvas.value_list))
            self.lossCanvas.draw_enable_flag = True

        if callback_dict.has_key('temp_error_list'):
            temp_error_list = callback_dict['temp_error_list']
            self.errorResultLabel.setText('Error: %f' % temp_error_list[-1])

            self.errorCanvas.draw_enable_flag = False
            if len(self.errorCanvas.index_list) == 0:
                for i in range(len(temp_error_list)):
                    self.errorCanvas.index_list.append(i)
            else:
                for i in range(len(temp_error_list)):
                    self.errorCanvas.index_list.append(self.errorCanvas.index_list[-1] + 1 + i)
            print(len(self.errorCanvas.index_list))
            for error in temp_error_list:
                self.errorCanvas.value_list.append(error)
            print(len(self.errorCanvas.value_list))
            self.errorCanvas.draw_enable_flag = True

        print('show chart use time: %f' % (time.time() - t))
        # 打开train的callback使能
        self.model.set_callback_enable(True)
        # 恢复按钮
        # self.button.setDisabled(False)

    # 初始化菜单栏
    def init_weight_menu(self, weight_list):
        for weight_index, weight in enumerate(weight_list):
            weight_button = MenuButton('&%s' % weight.name, self)
            weight_button.set_index(weight_index)
            # 此处用了两个信号，是为了解决自带信号 triggered() 不能带参数的问题
            self.connect(weight_button, QtCore.SIGNAL('triggered()'), weight_button.emit_f)
            self.connect(weight_button, QtCore.SIGNAL('clickMenuButtonWithWeightIndex(int)'), self.show_chart)
            self.weightMenu.addAction(weight_button)
            # 引用是为了后面恢复使能
            self.weightMenuItems.append(weight_button)

    def init_paras_labels(self):
        for item in dir(my_layer):
            if str(item).startswith('Layer_'):
                self.layerComboBox.insertItem(0, item)
        # 遍历的顺序为倒序，所以每次都插入到第一个
        self.layerComboBox.setCurrentIndex(0)

        for item in dir(my_loss):
            if str(item).startswith('loss_'):
                self.lossComboBox.insertItem(0, item)
        # 遍历的顺序为倒序，所以每次都插入到第一个
        self.lossComboBox.setCurrentIndex(0)

        for item in dir(my_optimizer):
            if str(item).startswith('optimizer_'):
                self.optimizerComboBox.insertItem(0, item)
        # 遍历的顺序为倒序，所以每次都插入到第一个
        self.optimizerComboBox.setCurrentIndex(0)