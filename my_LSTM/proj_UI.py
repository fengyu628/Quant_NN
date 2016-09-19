# coding:utf-8

from my_UI.my_MainWindow import MainWindow
from PyQt4 import QtGui
import sys

from my_LSTM.my_Model import MyRNNModel

# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ui = MainWindow(MyRNNModel)
    ui.show()
    sys.exit(app.exec_())