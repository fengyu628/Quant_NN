# coding:utf-8

from PyQt4 import QtGui, QtCore
from my_controls import MyButton, MyLabel, MyComboBox, MyEdit


class TrainFrame(QtGui.QFrame):
    def __init__(self, *args):
        super(TrainFrame, self).__init__(*args)

        self.trainingFilesButton = QtGui.QPushButton('Training Files')
        self.trainingFilesButton.setFont(QtGui.QFont("Calibri", 10))
        self.trainingFilesEdit = MyEdit(self)
        self.trainingFilesEdit.setDisabled(True)

        self.validateFilesButton = QtGui.QPushButton('Validate Files')
        self.validateFilesButton.setFont(QtGui.QFont("Calibri", 10))
        self.validateFilesEdit = MyEdit(self)
        self.validateFilesEdit.setDisabled(True)

        self.lossFunctionLabel = MyLabel('Loss:')
        self.lossComboBox = MyComboBox()

        self.optimizerLabel = MyLabel('Optimizer:')
        self.optimizerComboBox = MyComboBox()

        self.learningRateLabel = MyLabel('learning Rate:')
        self.learningRateEdit = MyEdit(self)

        self.epochLabel = MyLabel('Epoch:')
        self.epochEdit = MyEdit(self)

        self.trainButton = MyButton('Train', self)
        self.trainButton.setDisabled(True)

        self.pauseTrainButton = MyButton('Pause\nTrain', self)
        self.pauseTrainButton.setDisabled(True)

        self.resumeTrainButton = MyButton('Resume\nTrain', self)
        self.resumeTrainButton.setDisabled(True)

        self.stopTrainButton = MyButton('Stop\nTrain', self)
        self.stopTrainButton.setDisabled(True)

        grid = QtGui.QGridLayout()

        grid.addWidget(self.trainingFilesButton, 0, 0, 1, 1)
        grid.addWidget(self.trainingFilesEdit, 0, 1, 1, 1)
        grid.addWidget(self.validateFilesButton, 1, 0, 1, 1)
        grid.addWidget(self.validateFilesEdit, 1, 1, 1, 1)
        grid.addWidget(self.lossFunctionLabel, 2, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.lossComboBox, 2, 1, 1, 1)
        grid.addWidget(self.optimizerLabel, 3, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.optimizerComboBox, 3, 1, 1, 1)
        grid.addWidget(self.learningRateLabel, 4, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.learningRateEdit, 4, 1, 1, 1)
        grid.addWidget(self.epochLabel, 5, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.epochEdit, 5, 1, 1, 1)

        grid.addWidget(self.trainButton, 0, 2, 2, 1)
        grid.addWidget(self.pauseTrainButton, 0, 3, 2, 1)
        grid.addWidget(self.resumeTrainButton, 2, 2, 2, 1)
        grid.addWidget(self.stopTrainButton, 2, 3, 2, 1)

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