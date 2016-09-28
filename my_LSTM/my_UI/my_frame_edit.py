# coding:utf-8

from PyQt4 import QtGui
from PyQt4 import QtCore


class EditFrame(QtGui.QFrame):
    def __init__(self, *args):
        super(EditFrame, self).__init__(*args)

        # 初始化控件
        self.layerLabel = QtGui.QLabel('Layer Type:')
        self.layerComboBox = QtGui.QComboBox()

        self.inputDimLabel = QtGui.QLabel('Input Dimension:')
        self.inputDimEdit = QtGui.QLineEdit(self)
        # 暂时不许更改输入维度
        self.inputDimEdit.setDisabled(True)

        self.innerUnitsLabel = QtGui.QLabel('Inner Units:')
        self.innerUnitsEdit = QtGui.QLineEdit(self)

        self.lossFunctionLabel = QtGui.QLabel('Loss Function:')
        self.lossComboBox = QtGui.QComboBox()

        self.optimizerLabel = QtGui.QLabel('Optimizer Function:')
        self.optimizerComboBox = QtGui.QComboBox()

        self.learningRateLabel = QtGui.QLabel('learning Rate:')
        self.learningRateEdit = QtGui.QLineEdit(self)

        self.epochLabel = QtGui.QLabel('Epoch:')
        self.epochEdit = QtGui.QLineEdit(self)

        grid = QtGui.QGridLayout()
        grid.addWidget(self.layerLabel, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.layerComboBox, 0, 1, 1, 1)
        grid.addWidget(self.inputDimLabel, 1, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.inputDimEdit, 1, 1, 1, 1)
        grid.addWidget(self.innerUnitsLabel, 2, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.innerUnitsEdit, 2, 1, 1, 1)
        grid.addWidget(self.lossFunctionLabel, 3, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.lossComboBox, 3, 1, 1, 1)
        grid.addWidget(self.optimizerLabel, 4, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.optimizerComboBox, 4, 1, 1, 1)
        grid.addWidget(self.learningRateLabel, 5, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.learningRateEdit, 5, 1, 1, 1)
        grid.addWidget(self.epochLabel, 6, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.epochEdit, 6, 1, 1, 1)
        self.setLayout(grid)
        self.setFrameShape(QtGui.QFrame.StyledPanel)

    def build_model(self):
        # build 模型后，模型参数不再允许更改
        self.layerComboBox.setDisabled(True)
        self.inputDimEdit.setDisabled(True)
        self.innerUnitsEdit.setDisabled(True)

    def train_model(self):
        # 训练开始后，训练参数不再允许更改
        self.lossComboBox.setDisabled(True)
        self.optimizerComboBox.setDisabled(True)
        self.learningRateEdit.setDisabled(True)
        self.epochEdit.setDisabled(True)
