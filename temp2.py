# coding:utf-8
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QFileDialog

_fromUtf8 = QtCore.QString.fromUtf8

class MyWindow(QtGui.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.myButton = QtGui.QPushButton(self)
        self.myButton.setObjectName("myButton")
        self.myButton.setText("Test")
        self.myButton.clicked.connect(self.msg)

    def msg(self):
        # directory1 = QFileDialog.getExistingDirectory(self,
        #                                               "选取文件夹",
        #                                               "C:/")  # 起始路径
        # print(directory1)

        # fileName1 = QFileDialog.getOpenFileName(self,
        #                                                   "选取文件",
        #                                                   "C:/",
        #                                                   "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
        # print(fileName1)

        files = QFileDialog.getOpenFileNames(self,
                                                  "多文件选择",
                                                  "C:/",
                                                  "All Files (*);;Text Files (*.txt)")
        print(files)
        for file in files:
            print(file)

        # fileName2, ok2 = QFileDialog.getSaveFileName(self,
        #                                              "文件保存",
        #                                              "C:/",
        #                                              "All Files (*);;Text Files (*.txt)")


if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())    