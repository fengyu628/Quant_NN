# coding:utf-8

# import matplotlib.pyplot as plt
# from matplotlib.ticker import  MultipleLocator
from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
# import sys
import time
# import cv2.cv as cv
import copy
import pickle

from my_thread import TrainThread, MyGeneralThread
from my_chart import Chart
# from my_mplCanvas import MplCanvas
from my_frame_edit import EditFrame
from my_frame_button import ButtonFrame
from my_frame_canvas import CanvasFrame

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

        # ============================================= 窗口布局 =====================================================
        # self.setFixedSize(500, 300)
        self.resize(800, 600)
        self.setWindowTitle('Model')

        # 初始化菜单栏
        # 初始化 “File” 菜单
        menu_bar = self.menuBar()
        string_file = 'File'
        self.fileMenu = menu_bar.addMenu(string_file)

        self.openAction = QtGui.QAction('Open Model', self)
        self.openAction.setShortcut('Ctrl+O')
        self.openAction.setStatusTip('Open Model')
        self.connect(self.openAction, QtCore.SIGNAL('triggered()'), self.open_model)

        self.saveAction = QtGui.QAction('Save Model', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setStatusTip('Save Model')
        self.saveAction.setDisabled(True)
        self.connect(self.saveAction, QtCore.SIGNAL('triggered()'), self.save_model)

        self.exitAction = QtGui.QAction('Exit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.connect(self.exitAction, QtCore.SIGNAL('triggered()'), QtGui.qApp, QtCore.SLOT('quit()'))

        self.fileMenu.addAction(self.openAction)
        self.fileMenu.addAction(self.saveAction)
        self.fileMenu.addAction(self.exitAction)

        # 初始化 “Weight” 菜单
        self.weightMenu = menu_bar.addMenu('Weights')
        self.weightMenu.setDisabled(True)
        self.weightMenuItems = []

        self.editFrame = EditFrame()
        self.layerComboBox = self.editFrame.layerComboBox
        self.inputDimEdit = self.editFrame.inputDimEdit
        self.innerUnitsEdit = self.editFrame.innerUnitsEdit
        self.lossComboBox = self.editFrame.lossComboBox
        self.optimizerComboBox = self.editFrame.optimizerComboBox
        self.learningRateEdit = self.editFrame.learningRateEdit
        self.epochEdit = self.editFrame.epochEdit

        self.buttonFrame = ButtonFrame()
        self.buildButton = self.buttonFrame.buildButton
        self.trainButton = self.buttonFrame.trainButton
        self.pauseTrainButton = self.buttonFrame.pauseTrainButton
        self.resumeTrainButton = self.buttonFrame.resumeTrainButton
        self.stopTrainButton = self.buttonFrame.stopTrainButton
        self.closeAllChartsButton = self.buttonFrame.closeAllChartsButton
        self.connect(self.buildButton, QtCore.SIGNAL('clicked()'), self.build_model)
        self.connect(self.trainButton, QtCore.SIGNAL('clicked()'), self.train_model)
        self.connect(self.pauseTrainButton, QtCore.SIGNAL('clicked()'), self.pause_train)
        self.connect(self.resumeTrainButton, QtCore.SIGNAL('clicked()'), self.resume_train)
        self.connect(self.stopTrainButton, QtCore.SIGNAL('clicked()'), self.stop_train)
        self.connect(self.closeAllChartsButton, QtCore.SIGNAL('clicked()'), self.close_all_charts)

        bottom_frame = CanvasFrame()
        self.lossResultLabel = bottom_frame.lossResultLabel
        self.errorResultLabel = bottom_frame.errorResultLabel
        self.lossCanvas = bottom_frame.lossCanvas
        self.errorCanvas = bottom_frame.errorCanvas

        splitter_top = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter_top.addWidget(self.editFrame)
        splitter_top.addWidget(self.buttonFrame)

        splitter_vertical = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter_vertical.addWidget(splitter_top)
        splitter_vertical.addWidget(bottom_frame)

        self.setCentralWidget(splitter_vertical)

        # 初始化状态栏
        self.statusBar = QtGui.QStatusBar(self)
        self.statusBar.setObjectName('statusBar')
        self.setStatusBar(self.statusBar)
        self.statusRightLabel = QtGui.QLabel('')
        self.statusRightLabel.setAlignment(QtCore.Qt.AlignRight)
        self.statusBar.addPermanentWidget(self.statusRightLabel, 0)

        # self.setGeometry(300, 300, 300, 200)

        # =========================================== 结束窗口布局 ===========================================

        self.init_combo_box()

        # 训练模型的线程
        self.trainThread = TrainThread()
        # 连接子进程的信号和槽函数， 发射信号时所调用的函数
        self.trainThread.weights_updated_signal.connect(self.deal_with_train_callback)

        # 画 canvas 的线程
        self.drawCanvasThread = MyGeneralThread()
        self.drawCanvasThread.set_thread_function(self.draw_canvas)

        # 用于训练计时
        self.timer = QtCore.QTimer()
        self.connect(self.timer, QtCore.SIGNAL('timeout()'), self.timer_event)
        self.start_train_time = 0
        self.training_time = 0

        self.train_paused_flag = False
        self.stop_draw_canvas_thread_flag = False

        self.charts = []

        # 创建模型
        self.model = model()
        # 设置模型相关参数
        self.set_parameters_related_to_mode()

    # *********************************************** 消息处理函数 ******************************************************

    # 生成模型
    @QtCore.pyqtSlot()
    def build_model(self):
        try:
            # 设置模型参数
            self.model.layer_type = getattr(my_layer, str(self.layerComboBox.currentText()))
            self.model.input_dim = int(self.inputDimEdit.text())
            self.model.inner_units = int(self.innerUnitsEdit.text())
            self.model.build_layer()
        except Exception as e:
            print(e)
            QtGui.QMessageBox.warning(self, 'Build Error',
                                      str(e),
                                      QtGui.QMessageBox.Close)
            return
        
        self.set_status_before_train()

    # 开始训练模型
    @QtCore.pyqtSlot()
    def train_model(self):
        try:
            # 设置训练参数
            self.model.loss = getattr(my_loss, str(self.lossComboBox.currentText()))
            self.model.optimizer = getattr(my_optimizer, str(self.optimizerComboBox.currentText()))
            self.model.learning_rate = float(self.learningRateEdit.text())
            self.model.epoch = int(self.epochEdit.text())
        except Exception as e:
            print(e)
            QtGui.QMessageBox.warning(self, 'Train Error',
                                      str(e),
                                      QtGui.QMessageBox.Close)
            return

        self.editFrame.train_model()
        self.buttonFrame.train_model()

        # 启动线程
        self.trainThread.start(QtCore.QThread.HighPriority)
        self.stop_draw_canvas_thread_flag = False
        self.drawCanvasThread.start(QtCore.QThread.LowPriority)
        # 训练计时
        self.start_train_time = time.time()
        # 每0.1秒更新一次
        self.timer.start(100)

    # 暂停训练
    @QtCore.pyqtSlot()
    def pause_train(self):
        self.model.pause_training()
        self.train_paused_flag = True
        self.buttonFrame.pause_train()

    # 重新开始训练
    @QtCore.pyqtSlot()
    def resume_train(self):
        self.model.resume_training()
        self.train_paused_flag = False
        self.buttonFrame.resume_train()

    # 停止训练
    @QtCore.pyqtSlot()
    def stop_train(self):
        self.model.stop_training()
        self.timer.stop()
        self.stop_draw_canvas_thread_flag = True
        self.buttonFrame.stop_train()

    # 生成图表，并显示相应的权值
    @QtCore.pyqtSlot()
    def show_chart(self, weight_index):
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

    # 定时调用，用来显示训练时间
    @QtCore.pyqtSlot()
    def timer_event(self):
        if self.train_paused_flag is False:
            self.training_time += time.time() - self.start_train_time
            self.start_train_time = time.time()
        else:
            # 暂停训练时，时间不再累计
            self.start_train_time = time.time()
        # 在状态栏上显示训练时间
        self.statusRightLabel.setText(self.format_time_with_title(u'训练时间', self.training_time))

    # 导入模型
    @QtCore.pyqtSlot()
    def open_model(self):
        file_path = QtGui.QFileDialog.getOpenFileName(self,"Open Model","","pkl files(*.pkl)")
        if file_path == '':
            return
        with open(file_path, 'r') as f:
            self.stop_train()
            
            self.model = pickle.load(f)

            self.set_parameters_related_to_mode()

            self.set_status_before_train()

    # 保存模型
    @QtCore.pyqtSlot()
    def save_model(self):
        file_path =  QtGui.QFileDialog.getSaveFileName(self,'Save Model',"" ,"pkl files (*.pkl);;all files(*.*)")
        if file_path == '':
            return
        # np.savez(str(file_path), self.model.weights_list)
        with open(file_path, 'w') as f:
            model = copy.deepcopy(self.model)
            model.init_status_before_save()
            pickle.dump(model, f)

    # *************************************************** 其他函数 ******************************************************

    @staticmethod
    def format_time_with_title(title, time_seconds):
        return title + u'： %d天 %d小时 %d分 %d秒 .%d' % \
               (time_seconds / (24 * 60 * 60),
                time_seconds / (60 * 60),
                time_seconds / 60,
                time_seconds % 60, ((time_seconds % 1) * 10) % 10)

    # 画 canvas 的线程函数
    def draw_canvas(self):
        while True:
            # 防止空转时消耗CPU资源，做延时处理
            time.sleep(0.1)
            # 画 loss 曲线
            if self.lossCanvas.draw_enable_flag is True:
                x = copy.copy(self.lossCanvas.index_list)
                y = copy.copy(self.lossCanvas.value_list)
                if len(x) != len(y):
                    print(len(x), len(y))
                    continue
                self.lossCanvas.draw_data(x, y)
            # 画 error 曲线
            if self.errorCanvas.draw_enable_flag is True:
                x = copy.copy(self.errorCanvas.index_list)
                y = copy.copy(self.errorCanvas.value_list)
                if len(x) != len(y):
                    print(len(x), len(y))
                    continue
                self.errorCanvas.draw_data(x, y)
            # 训练结束，线程也结束
            if self.stop_draw_canvas_thread_flag is True:
                return

    # 处理callback传过来的权值
    def deal_with_train_callback(self, callback_dict):
        # 关闭train的callback使能
        self.model.set_callback_enable(False)
        # t = time.time()
        # 更新窗口中的权值
        for chart in self.charts:
            chart.show_weight(self.model.weights_list[chart.weight_index])

        if 'temp_loss_list' in callback_dict:
            temp_loss_list = callback_dict['temp_loss_list']
            self.lossResultLabel.setText('Loss: %f' % temp_loss_list[-1])

            self.lossCanvas.draw_enable_flag = False
            if len(self.lossCanvas.index_list) == 0:
                for i in range(len(temp_loss_list)):
                    self.lossCanvas.index_list.append(i)
            else:
                for i in range(len(temp_loss_list)):
                    self.lossCanvas.index_list.append(self.lossCanvas.index_list[-1] + 1 + i)
            # print(len(self.lossCanvas.index_list))
            for loss in temp_loss_list:
                self.lossCanvas.value_list.append(loss)
            # print(len(self.lossCanvas.value_list))
            self.lossCanvas.draw_enable_flag = True

        if 'temp_error_list' in callback_dict:
            temp_error_list = callback_dict['temp_error_list']
            self.errorResultLabel.setText('Error: %f' % temp_error_list[-1])

            self.errorCanvas.draw_enable_flag = False
            if len(self.errorCanvas.index_list) == 0:
                for i in range(len(temp_error_list)):
                    self.errorCanvas.index_list.append(i)
            else:
                for i in range(len(temp_error_list)):
                    self.errorCanvas.index_list.append(self.errorCanvas.index_list[-1] + 1 + i)
            # print(len(self.errorCanvas.index_list))
            for error in temp_error_list:
                self.errorCanvas.value_list.append(error)
            # print(len(self.errorCanvas.value_list))
            self.errorCanvas.draw_enable_flag = True

        if 'train_end' in callback_dict:
            if callback_dict['train_end'] is True:
                self.timer.stop()
                self.stop_draw_canvas_thread_flag = True
                self.statusRightLabel.setText(self.format_time_with_title(u'训练结束', self.training_time))
                self.buttonFrame.stop_train()
                QtGui.QMessageBox.information(self, 'Train Over',
                                              u'训练结束',
                                              QtGui.QMessageBox.Ok)

        # print('show chart use time: %f' % (time.time() - t))
        # 打开train的callback使能
        self.model.set_callback_enable(True)
        # 恢复按钮
        # self.button.setDisabled(False)

    # 初始化菜单栏
    def set_weight_menu(self, weight_list):
        self.weightMenu.clear()
        self.weightMenuItems = []
        for weight_index, weight in enumerate(weight_list):
            weight_button = MenuButton('&%s' % weight.name, self)
            weight_button.set_index(weight_index)
            weight_button.setStatusTip('Show "' + weight.name + '" in new window')
            # 此处用了两个信号，是为了解决自带信号 triggered() 不能带参数的问题
            self.connect(weight_button, QtCore.SIGNAL('triggered()'), weight_button.emit_f)
            self.connect(weight_button, QtCore.SIGNAL('clickMenuButtonWithWeightIndex(int)'), self.show_chart)
            self.weightMenu.addAction(weight_button)
            # 引用是为了后面恢复使能
            self.weightMenuItems.append(weight_button)
        # 使能菜单
        self.weightMenu.setDisabled(False)

    # 初始化下拉菜单
    def init_combo_box(self):
        for item in dir(my_layer):
            if str(item).startswith('Layer_'):
                # 遍历的顺序为倒序，所以每次都插入到第一个
                self.layerComboBox.insertItem(0, item)
        self.layerComboBox.setCurrentIndex(0)

        for item in dir(my_loss):
            if str(item).startswith('loss_'):
                # 遍历的顺序为倒序，所以每次都插入到第一个
                self.lossComboBox.insertItem(0, item)
        self.lossComboBox.setCurrentIndex(0)

        for item in dir(my_optimizer):
            if str(item).startswith('optimizer_'):
                # 遍历的顺序为倒序，所以每次都插入到第一个
                self.optimizerComboBox.insertItem(0, item)
        self.optimizerComboBox.setCurrentIndex(0)

    # 用于在改变模型后，设置与模型相关的参数
    def set_parameters_related_to_mode(self):
        self.inputDimEdit.setText(str(self.model.input_dim))
        self.innerUnitsEdit.setText(str(self.model.inner_units))
        self.learningRateEdit.setText(str(self.model.learning_rate))
        self.epochEdit.setText(str(self.model.epoch))
        # self.set_combo_box()
        for index in range(self.layerComboBox.count()):
            if self.model.layer_type.__name__ == self.layerComboBox.itemText(index):
                self.layerComboBox.setCurrentIndex(index)
        for index in range(self.lossComboBox.count()):
            if self.model.loss.__name__ == self.lossComboBox.itemText(index):
                self.lossComboBox.setCurrentIndex(index)
        for index in range(self.optimizerComboBox.count()):
            if self.model.optimizer.__name__ == self.optimizerComboBox.itemText(index):
                self.optimizerComboBox.setCurrentIndex(index)
        self.trainThread.set_model(self.model)

    # 用于点击“build”按钮，或者导入其他模型后，训练之前进行的操作
    def set_status_before_train(self):
        self.set_weight_menu(self.model.weights_list)
        # 设置界面
        self.editFrame.build_model()
        self.buttonFrame.build_model()
        # 使能保存模型选项
        self.saveAction.setDisabled(False)

        self.lossCanvas.index_list = []
        self.lossCanvas.value_list = []
        self.errorCanvas.index_list = []
        self.errorCanvas.value_list = []
