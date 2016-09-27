# coding:utf-8

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import time
import copy

class MplCanvas(FigureCanvas):
    """
    创建自己的画布
    """
    def __init__(self, title='Title'):

        # figsize 可以设置最大画布的大小
        self.fig = Figure(figsize=(20, 20))
        # self.fig = Figure()
        # self.fig = plt.gcf()
        # self.axes = plt.gca()
        self.ax = self.fig.add_subplot(111)
        # self.ax.spines['right'].set_color('none')
        # self.ax.spines['top'].set_color('none')
        # self.ax.spines['bottom'].set_color('none')
        # self.ax.spines['left'].set_color('none')
        self.title = title
        # self.ax.set_title(self.title, fontsize=14)
        self.ax.set_ylabel(self.title, fontsize=14)

        self.index_list = []
        self.value_list = []

        self.draw_enable_flag = True

        # self.subplot = plt.subplot()
        # self.ax = self.fig.add_subplot(111)
        # FigureCanvas.__init__(self, self.fig)
        super(MplCanvas, self).__init__(self.fig)
        # super(MplCanvas, self).__init__(self.axes)
        # FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)

    def draw_data(self, x, y):
        # while True:
        #     time.sleep(0.1)
        #     if self.draw_enable_flag is True:
        #         x = copy.copy(self.index_list)
        #         y = copy.copy(self.value_list)
        #         if len(x) != len(y):
        #             print(len(x), len(y))
        #             time.sleep(0.01)
        #             continue

            self.ax.clear()
            # self.ax.set_title(self.title, fontsize=14)
            # 不显示坐标
            # self.canvas.ax.set_xticks([])
            # self.canvas.ax.set_yticks([])
            # 设置刻度大小
            # xmajorLocator = MultipleLocator(0.1)
            # ymajorLocator = MultipleLocator(0.1)
            # self.canvas.ax.xaxis.set_major_locator(xmajorLocator)
            # self.canvas.ax.yaxis.set_major_locator(ymajorLocator)
            # 设置坐标范围
            # self.canvas.ax.set_xlim(-0.05, x_length*0.1 - 0.05)
            # self.canvas.ax.set_ylim(-0.05, y_length*0.1 - 0.05)
            # 设置标题
            # self.canvas.ax.set_title(weight_t.name, fontsize=14)
            self.ax.set_ylabel(self.title, fontsize=14)

            self.ax.plot(x, y, label="$sin(x)$", color="red", linewidth=1)
            self.draw()

