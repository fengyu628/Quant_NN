# coding:utf-8

from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
import cv2.cv as cv


'''
class MplCanvas(FigureCanvas):
    """
    创建自己的画布
    """
    def __init__(self, weight_shape):
        print(len(weight_shape))
        # 计算画布的尺寸
        scalar_factor = 5
        bias_factor = 0.
        if len(weight_shape) == 2:
            size = (float(weight_shape[1])/scalar_factor + bias_factor,
                    float(weight_shape[0])/scalar_factor + bias_factor)
        # weight为一维向量
        else:
            size = (float(weight_shape[0])/scalar_factor + bias_factor,
                   1.0 / scalar_factor + bias_factor)
        print('size: %s' % str(size))
        self.fig = Figure(figsize=size, dpi=100)
        # self.fig = Figure()
        # self.fig = plt.gcf()
        # self.axes = plt.gca()
        self.ax = self.fig.add_subplot(111)
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_color('none')
        self.ax.spines['left'].set_color('none')

        # self.subplot = plt.subplot()
        # self.ax = self.fig.add_subplot(111)
        # FigureCanvas.__init__(self, self.fig)
        super(MplCanvas, self).__init__(self.fig)
        # super(MplCanvas, self).__init__(self.axes)
        # FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)
'''


class Chart(QtGui.QWidget):
    """
    单独的窗口，用来显示一个权值矩阵
    """
    def __init__(self, 
                 weight_t, 
                 weight_index, 
                 image_scalar_factor=20,
                 x_gap=40,
                 y_gap=80,
                 max_width=600,
                 min_width=260,
                 parent=None):
        """
        :param weight_t: theano 格式的权值矩阵
        :param weight_index: weight_t 在 weight_list 中的下标
        :param image_scalar_factor: 每个权值在图片上占用的边长
        :param x_gap: 图片与窗口之间的空白量，宽度方向
        :param y_gap: 图片与窗口之间的空白量，高度方向
        :param max_width: 窗口宽度的最大值，如果窗口的宽度计算值大于此值，会压缩 image_scalar_factor，以保持窗口宽度不超过此参数
        :param min_width: 窗口宽度的最小值，如果窗口的宽度计算值小于此值，会保持窗口宽度等于此值，保证窗口的其他元素保持完整
        :param parent:
        """
        super(Chart, self).__init__(parent)

        # 返回当前的figure
        # self.canvas = MplCanvas(weight_shape)
        # layout = QtGui.QVBoxLayout(self)
        # layout.addWidget(self.canvas)

        self.weight_name = weight_t.name
        self.weight_index = weight_index
        self.weight_shape = weight_t.get_value().shape

        self.setWindowTitle(self.weight_name)

        self.title_label = QtGui.QLabel(self.weight_name)
        # self.title_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.title_label.setFont(QtGui.QFont("Calibri", 13))
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)

        self.image_label = QtGui.QLabel('weight_image')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.detail_label = QtGui.QLabel('detail_label')
        self.detail_label.setFont(QtGui.QFont("Calibri", 13))
        self.detail_label.setAlignment(QtCore.Qt.AlignCenter)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.title_label)
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.detail_label)
        self.setLayout(vbox)

        # 设置窗口大小
        self.image_scalar_factor = image_scalar_factor
        self.x_gap = x_gap
        self.y_gap = y_gap
        self.max_width = max_width
        self.min_width = min_width
        # weight 为权值矩阵
        if len(self.weight_shape) == 2:
            x_size = int(self.weight_shape[1]) * self.image_scalar_factor + self.x_gap
            y_size = int(self.weight_shape[0]) * self.image_scalar_factor + self.y_gap
            # 设置最大窗口宽度
            if x_size > self.max_width:
                x_size = self.max_width
                self.image_scalar_factor = (self.max_width - self.x_gap) / self.weight_shape[1]
                y_size = int(self.weight_shape[0]) * self.image_scalar_factor + self.y_gap
        # weight 为权值向量
        elif len(self.weight_shape) == 1:
            x_size = int(self.weight_shape[0]) * self.image_scalar_factor + self.x_gap
            y_size = 1 * self.image_scalar_factor + self.y_gap
            # 设置最大窗口宽度
            if x_size > self.max_width:
                x_size = self.max_width
                self.image_scalar_factor = (self.max_width - self.x_gap) / self.weight_shape[0]
        # weight 为权值标量
        else:
            x_size = 1 * self.image_scalar_factor + self.x_gap
            y_size = 1 * self.image_scalar_factor + self.y_gap

        # 设置最小窗口宽度
        if x_size < self.min_width:
            x_size = self.min_width

        print('x size:%d, y size:%d' % (x_size, y_size))
        self.setFixedSize(x_size, y_size)

    def show_weight(self, weight_t):
        """
        显示权值
        :param weight_t: theano 格式的权值矩阵
        :return:
        """
        weight = weight_t.get_value()
        # 权值矩阵
        if len(weight.shape) == 2:
            h = weight.shape[0]
            w = weight.shape[1]
            weight_array = np.asarray(weight)
        # 权值向量
        elif len(weight.shape) == 1:
            h = 1
            w = weight.shape[0]
            weight_array = np.asarray([weight])
        # 权值标量
        else:
            h = 1
            w = 1
            weight_array = np.asarray([[weight]])

        weight_min = np.min(weight_array)
        weight_max = np.max(weight_array)

        # 设置窗口显示的最大最小值
        detail_show = "min: %f   max: %f" % (weight_min, weight_max)
        self.detail_label.setText(detail_show)

        # 把权值转换成0到255之间的整数，用于图像显示
        weight_to_show = weight_array - weight_min
        scalar_factor = 255 / (weight_max - weight_min)
        array_float64 = (weight_to_show * scalar_factor) // 1
        # print(array_float64.min(), array_float64.max())
        # float64 转换成 int8
        array_uint8 = array_float64.astype(np.uint8)
        # np.ndarray 转换成 CvMat
        image_cvmat = cv.fromarray(array_uint8)
        # CvMat 转换成 IplImage
        image = cv.GetImage(image_cvmat)
        # 得到图片的宽和高
        # w, h = cv.GetSize(image)
        w_show = w * self.image_scalar_factor
        h_show = h * self.image_scalar_factor
        # 创建最终的图片，用于缩放之后的显示
        image_final = cv.CreateImage((w_show, h_show), 8, 1)
        # 缩放
        cv.Resize(image, image_final, interpolation=0)
        # 画外框
        line_type = cv.CV_AA
        cv.Rectangle(image_final, (-1, -1), (w_show, h_show), cv.RGB(0, 0, 0), 2, line_type, 0)
        # w, h = cv.GetSize(image_final)
        # self.image._imgData = image_final.tostring()
        self.image = QtGui.QImage(image_final.tostring(), w_show, h_show, QtGui.QImage.Format_Indexed8)
        # QImage 转换成 QPixmap
        pixmap = QtGui.QPixmap.fromImage(self.image)
        # 显示
        self.image_label.setPixmap(pixmap)

    # def close(self):
    #     print('close')
    #     super(Chart, self).close()

    def closeEvent(self, e):
        print('close %s' % self.weight_name)
        super(Chart, self).closeEvent(e)

        '''
        # 计算权值矩阵的尺寸
        if len(weight.shape) == 2:
            y_length = weight.shape[0]
            x_length = weight.shape[1]
        else:
            y_length = 1
            x_length = weight.shape[0]
        # x == [1,2,3,4,...,1,2,3,4...]
        x = np.asarray([i for i in range(x_length)] * y_length) * 0.1
        # y == [1,1,1,1,...,4,4,4,4,...]
        y = np.asarray([[i] * x_length for i in range(y_length)]).flatten() * 0.1
        self.canvas.ax.clear()
        # 不显示坐标
        # self.canvas.ax.set_xticks([])
        # self.canvas.ax.set_yticks([])
        # 设置刻度大小
        xmajorLocator = MultipleLocator(0.1)
        ymajorLocator = MultipleLocator(0.1)
        self.canvas.ax.xaxis.set_major_locator(xmajorLocator)
        self.canvas.ax.yaxis.set_major_locator(ymajorLocator)
        # 设置坐标范围
        self.canvas.ax.set_xlim(-0.05, x_length*0.1 - 0.05)
        self.canvas.ax.set_ylim(-0.05, y_length*0.1 - 0.05)
        # 设置标题
        self.canvas.ax.set_title(weight_t.name, fontsize=14)
        self.canvas.ax.set_xlabel("min:%f   max:%f" % (weight.min(), weight.max()), fontsize=14)
        self.canvas.ax.scatter(x, y, c=weight, s=200, alpha=0.4, marker='s', linewidths=1)
        self.canvas.draw()
        '''