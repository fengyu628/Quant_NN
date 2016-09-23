# coding:utf-8

import cv2.cv as cv
# import cv2
import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
_fromUtf8 = QString.fromUtf8

from PyQt4.QtCore import Qt

# -*- coding: utf-8 -*-
"""第一个程序"""
#from PyQt5 import QtWidgets
# from PyQt4.QtWidgets import *
from PyQt4.QtGui import QColor
import sys
class myDialog(QDialog):
    """docstring for myDialog"""
    def __init__(self, arg=None):
        super(myDialog, self).__init__(arg)
        self.setWindowTitle("first window")
        self.resize(400,100);
        addbtn=QPushButton(_fromUtf8('添加'))
        delbtn=QPushButton(_fromUtf8('清空'))
        conLayout = QHBoxLayout()
        self.sexComboBox=QComboBox()
        self.sexComboBox.insertItem(0,_fromUtf8("男"))
        # self.sexComboBox.insertItem(0,self.tr("Man"))
        self.sexComboBox.insertItem(1,_fromUtf8("女"))
        conLayout.addWidget(self.sexComboBox)
        conLayout.addWidget(addbtn)
        conLayout.addWidget(delbtn)
        self.setLayout(conLayout)
        self.sexComboBox.currentIndexChanged.connect(self.comboxchange)
        addbtn.clicked.connect(self.additem)
        delbtn.clicked.connect(self.clearComboBox)
    def clearComboBox(self):
        #清空组合框
        self.sexComboBox.clear()
    def additem(self):
        #添加文本
        self.sexComboBox.addItem(_fromUtf8('测试数据'))
    def comboxchange(self):
        QMessageBox.warning(self,_fromUtf8("警告"),str(self.sexComboBox.currentIndex())+self.tr(':')+self.sexComboBox.currentText(),QMessageBox.Yes)
app = QApplication(sys.argv)
#全局设置QPushButton的背景样式
dlg = myDialog()
dlg.show()
dlg.exec_()
app.exit()