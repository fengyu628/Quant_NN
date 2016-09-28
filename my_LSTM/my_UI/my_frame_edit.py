# coding:utf-8

from PyQt4 import QtGui
from PyQt4 import QtCore
from my_controls import MyLabel, MyEdit, MyComboBox


class EditFrame(QtGui.QFrame):
    def __init__(self, *args):
        super(EditFrame, self).__init__(*args)

        # 初始化控件
        self.layerLabel = MyLabel('Layer Type:')
        self.layerComboBox = MyComboBox()

        self.inputDimLabel = MyLabel('Input Dimension:')
        self.inputDimEdit = MyEdit(self)
        # 暂时不许更改输入维度
        self.inputDimEdit.setDisabled(True)

        self.innerUnitsLabel = MyLabel('Inner Units:')
        self.innerUnitsEdit = MyEdit(self)

        self.lossFunctionLabel = MyLabel('Loss:')
        self.lossComboBox = MyComboBox()

        self.optimizerLabel = MyLabel('Optimizer:')
        self.optimizerComboBox = MyComboBox()

        self.learningRateLabel = MyLabel('learning Rate:')
        self.learningRateEdit = MyEdit(self)

        self.epochLabel = MyLabel('Epoch:')
        self.epochEdit = MyEdit(self)

        grid = QtGui.QGridLayout()

        grid.addWidget(self.layerLabel, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.layerComboBox, 0, 1, 1, 1)
        grid.addWidget(self.inputDimLabel, 1, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.inputDimEdit, 1, 1, 1, 1)
        grid.addWidget(self.innerUnitsLabel, 2, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.innerUnitsEdit, 2, 1, 1, 1)

        grid.addWidget(self.lossFunctionLabel, 0, 2, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.lossComboBox, 0, 3, 1, 1)
        grid.addWidget(self.optimizerLabel, 1, 2, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.optimizerComboBox, 1, 3, 1, 1)
        grid.addWidget(self.learningRateLabel, 2, 2, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.learningRateEdit, 2, 3, 1, 1)
        grid.addWidget(self.epochLabel, 3, 2, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.epochEdit, 3, 3, 1, 1)

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
