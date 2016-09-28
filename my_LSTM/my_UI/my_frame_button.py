# coding:utf-8

from PyQt4 import QtGui


class ButtonFrame(QtGui.QFrame):
    def __init__(self, *args):
        super(ButtonFrame, self).__init__(*args)

        self.buildButton = QtGui.QPushButton('Build Model', self)
        self.buildButton.setFixedSize(100, 50)

        self.trainButton = QtGui.QPushButton('Train', self)
        self.trainButton.setFixedSize(100, 50)
        self.trainButton.setDisabled(True)

        self.pauseTrainButton = QtGui.QPushButton('Pause Train', self)
        self.pauseTrainButton.setFixedSize(100, 50)
        self.pauseTrainButton.setDisabled(True)

        self.resumeTrainButton = QtGui.QPushButton('Resume Train', self)
        self.resumeTrainButton.setFixedSize(100, 50)
        self.resumeTrainButton.setDisabled(True)

        self.stopTrainButton = QtGui.QPushButton('Stop Train', self)
        self.stopTrainButton.setFixedSize(100, 50)
        self.stopTrainButton.setDisabled(True)

        self.closeAllChartsButton = QtGui.QPushButton('Close All\nWeight Charts', self)
        self.closeAllChartsButton.setFixedSize(100, 50)

        grid = QtGui.QGridLayout()
        grid.addWidget(self.buildButton, 0, 0, 1, 1)
        grid.addWidget(self.trainButton, 0, 1, 1, 1)
        grid.addWidget(self.pauseTrainButton, 1, 0, 1, 1)
        grid.addWidget(self.resumeTrainButton, 1, 1, 1, 1)
        grid.addWidget(self.stopTrainButton, 2, 0, 1, 1)
        grid.addWidget(self.closeAllChartsButton, 2, 1, 1, 1)
        self.setLayout(grid)
        self.setFrameShape(QtGui.QFrame.StyledPanel)

    def build_model(self):
        # 把按钮禁用掉
        self.buildButton.setDisabled(True)
        # 使能训练按钮
        self.trainButton.setDisabled(False)

    def train_model(self):
        # 把按钮禁用掉
        self.trainButton.setDisabled(True)
        # 使能停止训练按钮
        self.pauseTrainButton.setDisabled(False)
        self.stopTrainButton.setDisabled(False)

    def pause_train(self):
        self.pauseTrainButton.setDisabled(True)
        self.resumeTrainButton.setDisabled(False)

    def resume_train(self):
        self.pauseTrainButton.setDisabled(False)
        self.resumeTrainButton.setDisabled(True)

    def stop_train(self):
        self.pauseTrainButton.setDisabled(True)
        self.resumeTrainButton.setDisabled(True)
        self.stopTrainButton.setDisabled(True)
