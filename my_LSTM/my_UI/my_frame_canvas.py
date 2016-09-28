# coding:utf-8

from PyQt4 import QtGui
from PyQt4 import QtCore
from my_mplCanvas import MplCanvas


class CanvasFrame(QtGui.QFrame):
    def __init__(self, *args):
        super(CanvasFrame, self).__init__(*args)

        self.lossResultLabel = QtGui.QLabel('Loss:')
        self.errorResultLabel = QtGui.QLabel('Error:')

        self.lossCanvas = MplCanvas(title='Loss')
        self.errorCanvas = MplCanvas(title='Error')

        grid = QtGui.QGridLayout()
        grid.addWidget(self.lossResultLabel, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        grid.addWidget(self.errorResultLabel, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        grid.addWidget(self.lossCanvas, 1, 0, 1, 2, QtCore.Qt.AlignCenter)
        grid.addWidget(self.errorCanvas, 2, 0, 1, 2, QtCore.Qt.AlignCenter)
        self.setLayout(grid)
        self.setFrameShape(QtGui.QFrame.StyledPanel)
