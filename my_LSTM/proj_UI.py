# coding:utf-8

import sys

from PyQt4 import QtGui

from my_MainWindow import MainWindow
from my_Model import MyRNNModel

# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ui = MainWindow(MyRNNModel)
    ui.show()
    sys.exit(app.exec_())