# coding:utf-8

from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy as np
import cv2.cv as cv


class Chart(QtGui.QWidget):
    """
    单独的窗口，用来显示一个权值矩阵
    """
    def __init__(self, 
                 name,
                 content,
                 index,
                 image_scalar_factor=20,
                 x_gap=40,
                 y_gap=80,
                 max_width=600,
                 min_width=260,
                 parent=None):
        """
        :param name: 名称
        :param content: 内容
        :param index: content_t 在 content_list 中的下标
        :param image_scalar_factor: 每个权值在图片上占用的边长
        :param x_gap: 图片与窗口之间的空白量，宽度方向
        :param y_gap: 图片与窗口之间的空白量，高度方向
        :param max_width: 窗口宽度的最大值，如果窗口的宽度计算值大于此值，会压缩 image_scalar_factor，以保持窗口宽度不超过此参数
        :param min_width: 窗口宽度的最小值，如果窗口的宽度计算值小于此值，会保持窗口宽度等于此值，保证窗口的其他元素保持完整
        :param parent:
        """
        super(Chart, self).__init__(parent)

        self.name = name
        self.index = index
        self.shape = content.shape

        self.setWindowTitle(self.name)

        self.title_label = QtGui.QLabel(self.name)
        # self.title_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.title_label.setFont(QtGui.QFont("Calibri", 13))
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)

        self.image_label = QtGui.QLabel('content_image')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image = None

        self.detail_label = QtGui.QLabel('detail_label')
        self.detail_label.setFont(QtGui.QFont("Calibri", 13))
        self.detail_label.setAlignment(QtCore.Qt.AlignCenter)

        v_box = QtGui.QVBoxLayout()
        v_box.addWidget(self.title_label)
        v_box.addWidget(self.image_label)
        v_box.addWidget(self.detail_label)
        self.setLayout(v_box)

        # 设置窗口大小
        self.image_scalar_factor = image_scalar_factor
        self.x_gap = x_gap
        self.y_gap = y_gap
        self.max_width = max_width
        self.min_width = min_width
        # content 为权值矩阵
        if len(self.shape) == 2:
            x_size = int(self.shape[1]) * self.image_scalar_factor + self.x_gap
            # 设置最大窗口宽度
            if x_size > self.max_width:
                x_size = self.max_width
                self.image_scalar_factor = float(self.max_width - self.x_gap) / self.shape[1]
                print(self.image_scalar_factor)
            y_size = int(self.shape[0]) * self.image_scalar_factor + self.y_gap
        # content 为权值向量
        elif len(self.shape) == 1:
            x_size = int(self.shape[0]) * self.image_scalar_factor + self.x_gap
            # 设置最大窗口宽度
            if x_size > self.max_width:
                x_size = self.max_width
                self.image_scalar_factor = float(self.max_width - self.x_gap) / self.shape[0]
            y_size = 1 * self.image_scalar_factor + self.y_gap
        # content 为权值标量
        else:
            x_size = 1 * self.image_scalar_factor + self.x_gap
            y_size = 1 * self.image_scalar_factor + self.y_gap

        # 设置最小窗口宽度
        if x_size < self.min_width:
            x_size = self.min_width

        # print('x size:%d, y size:%d' % (x_size, y_size))
        self.setFixedSize(x_size, y_size)

    def show_content(self, content):
        """
        显示权值
        :param content: 需要显示的内容
        :return:
        """
        # 权值矩阵
        if len(content.shape) == 2:
            h = content.shape[0]
            w = content.shape[1]
            content_array = np.asarray(content)
        # 权值向量
        elif len(content.shape) == 1:
            h = 1
            w = content.shape[0]
            content_array = np.asarray([content])
        # 权值标量
        else:
            h = 1
            w = 1
            content_array = np.asarray([[content]])

        content_min = np.min(content_array)
        content_max = np.max(content_array)

        # 设置窗口显示的最大最小值
        detail_show = "min: %f   max: %f" % (content_min, content_max)
        self.detail_label.setText(detail_show)

        # 把权值转换成0到255之间的整数，用于图像显示
        content_to_show = content_array - content_min
        scalar_factor = 255 / (content_max - content_min)
        array_float64 = (content_to_show * scalar_factor) // 1
        # print(array_float64.min(), array_float64.max())
        # float64 转换成 int8
        array_uint8 = array_float64.astype(np.uint8)
        # np.ndarray 转换成 CvMat
        image_cvmat = cv.fromarray(array_uint8)
        # CvMat 转换成 IplImage
        image = cv.GetImage(image_cvmat)
        # 得到图片的宽和高
        # w, h = cv.GetSize(image)
        w_show = int(w * self.image_scalar_factor)
        h_show = int(h * self.image_scalar_factor)
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
        print('close %s' % self.name)
        self.emit(QtCore.SIGNAL('closeChartWithIndex(int)'), self.index)
        super(Chart, self).closeEvent(e)
