# coding:utf-8

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as figureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import sys

class Example1(QWidget):
    def __init__(self,parent=None):
        super(Example1,self).__init__(parent)
        # 返回当前的figure
        figure = plt.gcf()
        self.canvas = figureCanvas(figure)
        x = [1,2,3]
        y = [4,5,6]
        plt.plot(x,y)
        plt.title('Example')
        plt.xlabel('x')
        plt.ylabel('y')
        self.canvas.draw()
        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)


class Example2(QWidget):
    def __init__(self, parent=None):
        super(Example2, self).__init__(parent)

        figure1 = plt.figure(1)  # 返回当前的figure
        x = [1, 2, 3]
        y = [4, 5, 6]
        plt.plot(x, y)
        plt.title('Example1')
        plt.xlabel('x')
        plt.ylabel('y')

        figure2 = plt.figure(2)  # 返回当前的figure
        x = [7, 8, 9]
        y = [4, 5, 6]
        plt.plot(x, y)
        plt.title('Example2')
        plt.xlabel('x')
        plt.ylabel('y')

        self.canvas1 = figureCanvas(figure1)
        self.canvas2 = figureCanvas(figure2)
        self.canvas1.draw()
        self.canvas2.draw()
        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas1)
        layout.addWidget(self.canvas2)


class Example3(QWidget):
    def __init__(self, parent=None):
        super(Example3, self).__init__(parent)

        figure = plt.figure(figsize=(10, 60), facecolor='green', edgecolor='red')
        # figsize = (8,4)表示figure的大小，屏幕显示 640 * 320 ， 输出显示 800*400，这个要注意。
        # 显示色和外框线条颜色设置。
        self.canvas = figureCanvas(figure)

        plt.subplot(211)  # 子区，2行，2列
        x = [1, 2, 3]
        y = [4, 5, 6]
        plt.plot(x, y)
        plt.title('Example')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(223)  # 子区，2行，2列
        x = [1, 2, 3]
        y = [4, 5, 6]
        plt.bar(x, y)
        plt.title('Example')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(224)  # 子区，2行，2列
        x = [1, 2, 3]
        y = [4, 5, 6]
        plt.scatter(x, y)
        plt.title('Example')
        plt.xlabel('x')
        plt.ylabel('y')

        self.canvas.draw()
        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)


class Example4(QWidget):
    def __init__(self, parent=None):
        super(Example4, self).__init__(parent)
        figure = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
        canvas = figureCanvas(figure)
        x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
        c = np.cos(x)
        s = np.sin(x)
        plt.plot(x, c, color='blue', linewidth=1.0, linestyle='-', label='$cos(x)$')  # 设置颜色，线条的格式和粗细
        plt.plot(x, s, color='green', linewidth=1.0, linestyle='-', label='$sin(x)$')

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))

        plt.xlim(x.min() * 1.1, x.max() * 1.1)  # X轴的范围
        plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])  # X轴的刻度值
        plt.ylim(s.min() * 1.1, s.max() * 1.1)  # Y轴的范围
        plt.yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$+1$'])  # 设置Y轴的刻度值,第二个参数对其进行格式化

        # 添加注释和箭头以及虚线
        t = np.pi * 2 / 3
        plt.plot([t, t], [0, np.sin(t)], color='red', linewidth=2.5, linestyle='--')
        plt.scatter([t], [np.sin(t)], 50, color='red')  # 50代表散点的大小，应该是像素值

        plt.plot([t, t], [0, np.cos(t)], color='green', linewidth=2.5, linestyle='--')
        plt.scatter([t], [np.cos(t)], 50, color='green')

        plt.annotate(r'$sin(\frac{2\pi}{3})=(\frac{\sqrt{3}}{2})$',
                     xy=(t, np.sin(t)), xycoords='data',
                     xytext=(10, 30), textcoords='offset points', fontsize=16,
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.1'))
        plt.annotate(r'$cos(\frac{2\pi}{3})=(\frac{\sqrt{3}}{2})$',
                     xy=(t, np.cos(t)), xycoords='data',
                     xytext=(-120, -30), textcoords='offset points', fontsize=16,
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.1'))  # 后面的参数应该是角度，类似于偏离度，1的时候接近垂直
        plt.legend(loc='upper left')

        for i in ax.get_xticklabels() + ax.get_yticklabels():
            i.set_fontsize(15)
            i.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.65))

        canvas.draw()
        layout = QHBoxLayout(self)
        layout.addWidget(canvas)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Example4()
    ui.show()
    app.exec_()

# ================================================================================================================

class MplCanvas(FigureCanvas):
    """
    Creates a canvas on which to draw our widgets
    """

    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class marketdephWidget(QtGui.QWidget):
    """
    The market deph graph
    """

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


    # 绘图函数（在按下按钮的符文）：
    # initialize the mplwidgets
    def PlotFunc(self):
        randomNumbers = random.sample(range(0, 10), 10)
        self.ui.widget.canvas.ax.clear()
        self.ui.widget.canvas.ax.plot(randomNumbers)
        self.ui.widget.canvas.draw()