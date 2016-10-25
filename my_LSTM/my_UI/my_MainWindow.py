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
from my_frame_model import ModelFrame
from my_frame_train import TrainFrame
from my_frame_canvas import CanvasFrame

from my_LSTM import my_layer
from my_LSTM import my_loss
from my_LSTM import my_optimizer
from my_LSTM.my_data_processor import csv_file_to_train_data

# _fromUtf8 = QtCore.QString.fromUtf8
canvas_show_max_length = 500


class MenuButton(QtGui.QAction):
    """
    重写菜单栏的项目。
    用来实现，在被触发时，发射带参数的信号，而且这个参数是和该项目绑定的。
    """
    def __init__(self, *args):
        super(MenuButton, self).__init__(*args)
        self.name = None
        self.index = 0

    # 这个 index 和 weight_list 、grad_list 的顺序相对应，用于调取相应的权值和梯度
    def set_index(self, index):
        self.index = index

    # 发射带有序号的信号
    def emit_f(self):
        self.emit(QtCore.SIGNAL('clickMenuButtonWithIndex(int)'), self.index)
        self.setDisabled(True)


class MainWindow(QtGui.QMainWindow):
    """
    主窗口
    """
    def __init__(self, model, parent=None):
        super(MainWindow, self).__init__(parent)

        # ============================================= 窗口布局 =====================================================
        # self.setFixedSize(500, 300)
        self.resize(800, 650)
        self.setWindowTitle('Model')

        # 初始化菜单栏
        # 初始化 “File” 菜单
        menu_bar = self.menuBar()
        self.fileMenu = menu_bar.addMenu('File')

        self.exitAction = QtGui.QAction('Exit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.connect(self.exitAction, QtCore.SIGNAL('triggered()'), QtGui.qApp, QtCore.SLOT('quit()'))

        self.fileMenu.addAction(self.exitAction)

        # 初始化 “Weight” 菜单
        self.weightMenu = menu_bar.addMenu('Weights')
        self.weightMenu.setDisabled(True)
        self.weightMenuItems = []

        # 初始化 “Grads” 菜单
        self.gradMenu = menu_bar.addMenu('Grads')
        self.gradMenu.setDisabled(True)
        self.gradMenuItems = []

        self.ModelFrame = ModelFrame()
        self.layerComboBox = self.ModelFrame.layerComboBox
        self.inputDimEdit = self.ModelFrame.inputDimEdit
        self.innerUnitsEdit = self.ModelFrame.innerUnitsEdit
        self.parametersEdit = self.ModelFrame.parametersEdit
        self.buildButton = self.ModelFrame.buildButton
        self.loadButton = self.ModelFrame.loadButton
        self.saveButton = self.ModelFrame.saveButton

        self.TrainFrame = TrainFrame()
        self.trainingFilesButton = self.TrainFrame.trainingFilesButton
        self.trainingFilesEdit = self.TrainFrame.trainingFilesEdit
        self.validateFilesButton = self.TrainFrame.validateFilesButton
        self.validateFilesEdit = self.TrainFrame.validateFilesEdit
        self.lossComboBox = self.TrainFrame.lossComboBox
        self.optimizerComboBox = self.TrainFrame.optimizerComboBox
        self.batchSizeEdit = self.TrainFrame.batchSizeEdit
        self.learningRateEdit = self.TrainFrame.learningRateEdit
        self.epochEdit = self.TrainFrame.epochEdit
        self.trainButton = self.TrainFrame.trainButton
        self.pauseTrainButton = self.TrainFrame.pauseTrainButton
        self.resumeTrainButton = self.TrainFrame.resumeTrainButton
        self.stopTrainButton = self.TrainFrame.stopTrainButton

        self.connect(self.buildButton, QtCore.SIGNAL('clicked()'), self.build_model)
        self.connect(self.loadButton, QtCore.SIGNAL('clicked()'), self.open_model)
        self.connect(self.saveButton, QtCore.SIGNAL('clicked()'), self.save_model)
        self.connect(self.trainingFilesButton, QtCore.SIGNAL('clicked()'), self.training_files)
        self.connect(self.validateFilesButton, QtCore.SIGNAL('clicked()'), self.validate_files)
        self.connect(self.trainButton, QtCore.SIGNAL('clicked()'), self.train_model)
        self.connect(self.pauseTrainButton, QtCore.SIGNAL('clicked()'), self.pause_train)
        self.connect(self.resumeTrainButton, QtCore.SIGNAL('clicked()'), self.resume_train)
        self.connect(self.stopTrainButton, QtCore.SIGNAL('clicked()'), self.stop_train)

        bottom_frame = CanvasFrame()
        self.lossResultLabel = bottom_frame.lossResultLabel
        self.errorResultLabel = bottom_frame.errorResultLabel
        self.lossCanvas = bottom_frame.lossCanvas
        self.errorCanvas = bottom_frame.errorCanvas

        splitter_top = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter_top.addWidget(self.ModelFrame)
        splitter_top.addWidget(self.TrainFrame)

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

        self.weight_charts = []
        self.grad_charts = []

        # 创建模型
        self.model = model()
        # 设置模型相关参数
        self.set_parameters_related_to_mode()

        # TODO:----------- 调试 -----------
        # self.build_model()

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

        # TODO:----------- 调试 -----------
        # self.training_files()
        # self.validate_files()

    # 开始训练模型
    @QtCore.pyqtSlot()
    def train_model(self):
        try:
            # 设置训练参数
            self.model.loss = getattr(my_loss, str(self.lossComboBox.currentText()))
            self.model.optimizer = getattr(my_optimizer, str(self.optimizerComboBox.currentText()))
            self.model.learning_rate = float(self.learningRateEdit.text())
            self.model.mini_batch_size = float(self.batchSizeEdit.text())
            self.model.epoch = int(self.epochEdit.text())
        except Exception as e:
            print(e)
            QtGui.QMessageBox.warning(self, 'Train Error',
                                      str(e),
                                      QtGui.QMessageBox.Close)
            return

        self.TrainFrame.train_model()

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
        self.TrainFrame.pause_train()

    # 重新开始训练
    @QtCore.pyqtSlot()
    def resume_train(self):
        self.model.resume_training()
        self.train_paused_flag = False
        self.TrainFrame.resume_train()

    # 停止训练
    @QtCore.pyqtSlot()
    def stop_train(self):
        self.model.stop_training()
        self.timer.stop()
        self.stop_draw_canvas_thread_flag = True
        self.TrainFrame.stop_train()

    # 生成图表，并显示相应的权值
    @QtCore.pyqtSlot()
    def show_weight_chart(self, weight_index):
        weight_chart = Chart((self.model.weights_list[weight_index]).name,
                             (self.model.weights_list[weight_index]).get_value(),
                             weight_index)
        self.connect(weight_chart, QtCore.SIGNAL('closeChartWithIndex(int)'), self.close_weight_chart_event)
        self.weight_charts.append(weight_chart)
        weight_chart.show_content((self.model.weights_list[weight_index]).get_value())
        weight_chart.show()

    # 生成图表，并显示相应的梯度
    @QtCore.pyqtSlot()
    def show_grad_chart(self, grad_index):
        chart_name = (self.model.weights_list[grad_index]).name + ' Grad'
        grad_chart = Chart(chart_name,
                           (self.model.grads_list[grad_index]).get_value(),
                           grad_index)
        self.connect(grad_chart, QtCore.SIGNAL('closeChartWithIndex(int)'), self.close_grad_chart_event)
        self.grad_charts.append(grad_chart)
        grad_chart.show_content((self.model.grads_list[grad_index]).get_value())
        grad_chart.show()

    # 有 Weight Chart 被关闭的事件处理
    @QtCore.pyqtSlot()
    def close_weight_chart_event(self, weight_index):
        print('close weight chart of index: %d' % weight_index)
        for chart in self.weight_charts:
            if chart.index == weight_index:
                self.weight_charts.remove(chart)
                # 恢复权值菜单元素的使能
                for item in self.weightMenuItems:
                    if item.index == weight_index:
                        item.setDisabled(False)
        print('weight charts count:%d' % len(self.weight_charts))

    # 有 Grad Chart 被关闭的事件处理
    @QtCore.pyqtSlot()
    def close_grad_chart_event(self, grad_index):
        print('close grad chart of index: %d' % grad_index)
        for chart in self.grad_charts:
            if chart.index == grad_index:
                self.grad_charts.remove(chart)
                # 恢复权值菜单元素的使能
                for item in self.gradMenuItems:
                    if item.index == grad_index:
                        item.setDisabled(False)
        print('grad charts count:%d' % len(self.grad_charts))

    # 关闭所有 Weight Chart
    @QtCore.pyqtSlot()
    def close_all_weight_charts(self):
        self.weight_charts = []
        for item in self.weightMenuItems:
            item.setDisabled(False)

    # 关闭所有 grad Chart
    @QtCore.pyqtSlot()
    def close_all_grad_charts(self):
        self.grad_charts = []
        for item in self.gradMenuItems:
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
        self.statusRightLabel.setText(self.format_time_with_title(u'训练用时', self.training_time))

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
            model.set_status_before_save()
            pickle.dump(model, f)

    @QtCore.pyqtSlot()
    def training_files(self):
        # files = QtGui.QFileDialog.getOpenFileNames(self,
        #                                            "Select training files",
        #                                            "..",
        #                                            "CSV Files (*.csv);;All Files (*)")
        files = ['..\\training_files\\fu02_20081203.csv']
        # 每次选择文件时，都清空数组，重新生成
        self.model.train_x = []
        self.model.train_y = []
        for index, f in enumerate(files):
            # 从文件生成训练数据
            x_array, y_array = csv_file_to_train_data(f)
            print(x_array.shape, y_array.shape)
            if len(self.model.train_x) == 0:
                self.model.train_x = x_array
            else:
                self.model.train_x = np.append(self.model.train_x, x_array, axis=0)
            if len(self.model.train_y) == 0:
                self.model.train_y = y_array
            else:
                self.model.train_y = np.append(self.model.train_y, y_array, axis=0)
        print(np.asarray(self.model.train_x).shape, np.asarray(self.model.train_y).shape)
        self.trainingFilesEdit.setText('%d files,  %d train data' % (len(files), len(self.model.train_x)))

    @QtCore.pyqtSlot()
    def validate_files(self):
        # files = QtGui.QFileDialog.getOpenFileNames(self,
        #                                            "Select validate files",
        #                                            "..",
        #                                            "CSV Files (*.csv);;All Files (*)")
        files = ['..\\training_files\\fu02_20081203.csv']
        # 每次选择文件时，都清空数组，重新生成
        self.model.validate_x = []
        self.model.validate_y = []
        for index, f in enumerate(files):
            # 从文件生成训练数据
            x_array, y_array = csv_file_to_train_data(f)
            print(x_array.shape, y_array.shape)
            if len(self.model.validate_x) == 0:
                self.model.validate_x = x_array
            else:
                self.model.validate_x = np.append(self.model.validate_x, x_array, axis=0)
            if len(self.model.validate_y) == 0:
                self.model.validate_y = y_array
            else:
                self.model.validate_y = np.append(self.model.validate_y, y_array, axis=0)
        print(np.asarray(self.model.validate_x).shape, np.asarray(self.model.validate_y).shape)
        self.validateFilesEdit.setText('%d files,  %d validate data' % (len(files), len(self.model.validate_x)))

    # *************************************************** 其他函数 ******************************************************

    @staticmethod
    def format_time_with_title(title, time_seconds):
        return title + u'： %d天 %d小时 %d分 %d秒 .%d' % \
               (time_seconds / (24 * 60 * 60),
                (time_seconds / (60 * 60)) % 24,
                (time_seconds / 60) % 60,
                time_seconds % 60,
                ((time_seconds % 1) * 10) % 10)

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
        # 使能梯度菜单
        self.gradMenu.setDisabled(False)
        # t = time.time()
        # 更新窗口中的权值
        for chart in self.weight_charts:
            chart.show_content((self.model.weights_list[chart.index]).get_value())
        for chart in self.grad_charts:
            chart.show_content((self.model.grads_list[chart.index]).get_value())

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

            # 仅仅保留最后的 canvas_show_max_length 个数据，用于显示
            if len(self.lossCanvas.index_list) > canvas_show_max_length:
                self.lossCanvas.index_list = self.lossCanvas.index_list[-canvas_show_max_length:]
                self.lossCanvas.value_list = self.lossCanvas.value_list[-canvas_show_max_length:]

            # print('-----------', self.lossCanvas.index_list[-1])
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

            # 仅仅保留最后的 canvas_show_max_length 个数据，用于显示
            if len(self.errorCanvas.index_list) > canvas_show_max_length:
                self.errorCanvas.index_list = self.errorCanvas.index_list[-canvas_show_max_length:]
                self.errorCanvas.value_list = self.errorCanvas.value_list[-canvas_show_max_length:]

            self.errorCanvas.draw_enable_flag = True

        if 'train_end' in callback_dict:
            if callback_dict['train_end'] is True:
                self.timer.stop()
                self.stop_draw_canvas_thread_flag = True
                self.statusRightLabel.setText(self.format_time_with_title(u'训练结束', self.training_time))
                self.TrainFrame.stop_train()
                QtGui.QMessageBox.information(self, 'Train Over',
                                              u'训练结束',
                                              QtGui.QMessageBox.Ok)

        # print('show chart use time: %f' % (time.time() - t))
        # 打开train的callback使能
        self.model.set_callback_enable(True)
        # 恢复按钮
        # self.button.setDisabled(False)

    # 初始化权值菜单
    def set_weight_menu(self, weight_list):
        self.weightMenu.clear()
        self.weightMenuItems = []
        for weight_index, weight in enumerate(weight_list):
            weight_button = MenuButton('&%s' % weight.name, self)
            weight_button.set_index(weight_index)
            weight_button.setStatusTip('Show "' + weight.name + '" in new window')
            # 此处用了两个信号，是为了解决自带信号 triggered() 不能带参数的问题
            self.connect(weight_button,
                         QtCore.SIGNAL('triggered()'),
                         weight_button.emit_f)
            self.connect(weight_button,
                         QtCore.SIGNAL('clickMenuButtonWithIndex(int)'),
                         self.show_weight_chart)
            self.weightMenu.addAction(weight_button)
            # 引用是为了后面恢复使能
            self.weightMenuItems.append(weight_button)
        self.weightMenu.addSeparator()
        close_all_weight_charts_button = QtGui.QAction('close all weight charts', self)
        close_all_weight_charts_button.setStatusTip('Close all weight charts')
        self.connect(close_all_weight_charts_button, QtCore.SIGNAL('triggered()'), self.close_all_weight_charts)
        self.weightMenu.addAction(close_all_weight_charts_button)
        # 使能菜单
        self.weightMenu.setDisabled(False)

    #
    def set_grad_menu(self, weight_list):
        self.gradMenu.clear()
        self.gradMenuItems = []
        for weight_index, weight in enumerate(weight_list):
            grad_button = MenuButton('&%s Grad' % weight.name, self)
            grad_button.set_index(weight_index)
            grad_button.setStatusTip('Show grad of "' + weight.name + '" in new window')
            # 此处用了两个信号，是为了解决自带信号 triggered() 不能带参数的问题
            self.connect(grad_button,
                         QtCore.SIGNAL('triggered()'),
                         grad_button.emit_f)
            self.connect(grad_button,
                         QtCore.SIGNAL('clickMenuButtonWithIndex(int)'),
                         self.show_grad_chart)
            self.gradMenu.addAction(grad_button)
            # 引用是为了后面恢复使能
            self.gradMenuItems.append(grad_button)
        self.gradMenu.addSeparator()
        close_all_grad_charts_button = QtGui.QAction('close all grad charts', self)
        close_all_grad_charts_button.setStatusTip('Close all grad charts')
        self.connect(close_all_grad_charts_button, QtCore.SIGNAL('triggered()'), self.close_all_grad_charts)
        self.gradMenu.addAction(close_all_grad_charts_button)

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
        self.batchSizeEdit.setText(str(self.model.mini_batch_size))
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
        self.set_grad_menu(self.model.weights_list)
        # 计算模型的参数数量
        parameters_count = 0
        for w in self.model.weights_list:
            parameters_count += w.get_value().size
        self.parametersEdit.setText(str(parameters_count))
        # 设置界面
        self.ModelFrame.build_model()
        self.TrainFrame.build_model()
        # 使能保存模型选项
        self.saveButton.setDisabled(False)
        # 清空 canvas 显示
        self.lossCanvas.index_list = []
        self.lossCanvas.value_list = []
        self.errorCanvas.index_list = []
        self.errorCanvas.value_list = []
