# coding:utf-8

import cv2.cv as cv
# import cv2
import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
_fromUtf8 = QString.fromUtf8

# np_gray = (np.random.rand(10, 20) * 255) // 1
# print(np_gray)
# # np_gray = np.int_(np_gray)
# # np_gray = np.float_(np_gray)
# gray = np_gray.astype(np.uint8)
# print(gray.dtype)
# print(gray)
#
# # exit(8)
#
# image_cvmat = cv.fromarray(gray)
# # print(image_cvmat)
#
# # image = cv.CreateImage((100, 100), 8, 1)
# image = cv.GetImage(image_cvmat)
# print(image)
# # image_np = np.asarray(image_color)
# # print(image_np)
# # cv.CvtColor(image_gray, image_color, cv.CV_GRAY2BGR)
#
# image_final = cv.CreateImage((220, 100), 8, 1)
# print(image_final)
#
# cv.Resize(image, image_final, interpolation=0)
#
# # cv.ShowImage("Focus", image_final)
# # cv.WaitKey(0)

# class OpenCVQImage(QImage):
#     def __init__(self, opencv_bgr_img):
#         depth, n_channels = opencv_bgr_img.depth, opencv_bgr_img.nChannels
#         if depth != cv.IPL_DEPTH_8U or n_channels != 1:
#             raise ValueError("the input image must be 8-bit, 1-channel")
#         w, h = cv.GetSize(opencv_bgr_img)
#         opencv_rgb_img = cv.CreateImage((w, h), depth, 3)
#         # it's assumed the image is in BGR format
#         cv.CvtColor(opencv_bgr_img, opencv_rgb_img, cv.CV_GRAY2RGB)
#         self._imgData = opencv_bgr_img.tostring()
#         super(OpenCVQImage, self).__init__(self._imgData, w, h,
#                                            QImage.Format_RGB888)

class Chart(QWidget):
    def __init__(self, parent=None):
        super(Chart, self).__init__(parent)
        # w, h = cv.GetSize(image_final)
        # self.image = QImage(image_final.tostring(), w, h, QImage.Format_Indexed8)
        # self.image = QImage()

        self.setFixedSize(500, 300)

        self.piclabel = QLabel('pic')
        self.btn = QPushButton(_fromUtf8('更新'), self)
        self.btn.setGeometry(QRect(215, 190, 80, 26))
        self.connect(self.btn, SIGNAL('clicked()'), self.change)
        vbox = QVBoxLayout()
        vbox.addWidget(self.piclabel)
        vbox.addWidget(self.btn)
        self.setLayout(vbox)
        self.change()

        # self.image._imgData = image_final.tostring()
        # pixmap = QPixmap.fromImage(self.image)
        # self.piclabel.setPixmap(pixmap)
        # self.show()

    @pyqtSlot()
    def change(self):
        np_gray = (np.random.rand(10, 20) * 255) // 1
        gray = np_gray.astype(np.uint8)
        image_cvmat = cv.fromarray(gray)
        image = cv.GetImage(image_cvmat)
        image_final = cv.CreateImage((220, 100), 8, 1)
        cv.Resize(image, image_final, interpolation=0)
        w, h = cv.GetSize(image_final)
        # self.image._imgData = image_final.tostring()
        self.image = QImage(image_final.tostring(), w, h, QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(self.image)
        self.piclabel.setPixmap(pixmap)

    # def paintEvent(self, e):
    #     painter = QtGui.QPainter(self)
    #     painter.drawImage(QtCore.QPoint(0, 0), OpenCVQImage(image_final))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Chart()
    ui.show()
    # ui.update()
    sys.exit(app.exec_())