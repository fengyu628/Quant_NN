# coding:utf-8

from PyQt4 import QtGui
from PyQt4 import QtCore
from my_controls import MyLabel, MyEdit, MyComboBox, MyButton


class ModelFrame(QtGui.QFrame):
    def __init__(self, *args):
        super(ModelFrame, self).__init__(*args)

        # 初始化控件
        self.layerLabel = MyLabel('Layer Type:')
        self.layerComboBox = MyComboBox()

        self.inputDimLabel = MyLabel('Input Dimension:')
        self.inputDimEdit = MyEdit(self)
        # 暂时不许更改输入维度
        self.inputDimEdit.setDisabled(True)

        self.innerUnitsLabel = MyLabel('Inner Units:')
        self.innerUnitsEdit = MyEdit(self)

        self.parametersLabel = MyLabel('Parameters:')
        self.parametersEdit = MyEdit(self)
        self.parametersEdit.setDisabled(True)

        self.buildButton = MyButton('Build\nModel', self)

        self.loadButton = MyButton('Load\nModel', self)

        self.saveButton = MyButton('Save\nModel', self)
        self.saveButton.setDisabled(True)

        grid = QtGui.QGridLayout()

        grid.addWidget(self.layerLabel, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.layerComboBox, 0, 1, 1, 1)
        grid.addWidget(self.inputDimLabel, 1, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.inputDimEdit, 1, 1, 1, 1)
        grid.addWidget(self.innerUnitsLabel, 2, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.innerUnitsEdit, 2, 1, 1, 1)
        grid.addWidget(self.parametersLabel, 3, 0, 1, 1, QtCore.Qt.AlignRight)
        grid.addWidget(self.parametersEdit, 3, 1, 1, 1)

        grid.addWidget(self.buildButton, 0, 2, 2, 1)
        grid.addWidget(self.loadButton, 2, 2, 2, 1)
        grid.addWidget(self.saveButton, 2, 3, 2, 1)

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
