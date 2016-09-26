# coding:utf-8

from PyQt4 import QtCore


class TrainThread(QtCore.QThread):
    """
    训练的模型的线程，因为训练时会阻塞主程序，故新起一个线程。
    """
    # 声明一个信号，同时返回一个list，同理什么都能返回啦
    # weights_updated_signal = QtCore.pyqtSignal(list)
    weights_updated_signal = QtCore.pyqtSignal(dict)

    def __init__(self, model, parent=None):
        super(TrainThread, self).__init__(parent)
        # 添加模型的引用
        self.model = model

    # 实例的start()调用
    def run(self):
        # 设置模型训练时的回调函数。回调函数为发射信号，参数是list型式（也就是权值列表）
        self.model.set_callback_when_weight_updated(self.weights_updated_signal.emit)
        self.model.train()


class MyGeneralThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(MyGeneralThread, self).__init__(parent)
        self.function = None

    def set_thread_function(self, function):
        self.function = function

    def run(self):
        self.function()