# coding:utf-8

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import PyQt4.QtGui as QtGui
from numpy import random
import sys
import matplotlib.pyplot as plt
import numpy as np


class MplCanvas(FigureCanvas):
    """
    Creates a canvas on which to draw our widgets
    """
    def __init__(self):
        # self.fig = Figure()
        self.fig = plt.gcf()
        # self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        # FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)

class Example1(QWidget):
    def __init__(self,parent=None):
        super(Example1,self).__init__(parent)
        # 返回当前的figure
        # figure = plt.gcf()
        self.canvas = MplCanvas()
        n = 10
        x = np.asarray([i for i in range(n)] * n) * 0.1
        y = np.asarray([[i] * n for i in range(n)]).flatten() * 0.1
        # color = range(n ** 2)
        color = np.random.random(n ** 2)
        plt.scatter(x, y, c=color, s=1000, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()
        self.button = QtGui.QPushButton('Button', self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)

        self.connect(self.button, QtCore.SIGNAL('clicked()'), self, QtCore.SLOT("change()"))

    @QtCore.pyqtSlot()
    def change(self):
        print('change')
        self.canvas.fig.clear()
        n = 10
        x = np.asarray([i for i in range(n)] * n) * 0.1
        y = np.asarray([[i] * n for i in range(n)]).flatten() * 0.1
        # color = range(n ** 2)
        color = np.random.random(n ** 2)
        plt.scatter(x, y, c=color, s=1000, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Example1()
    ui.show()
    app.exec_()