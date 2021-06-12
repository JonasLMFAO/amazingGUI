from typing import NewType
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2
from PyQt5.QtCore import *
import numpy as np
from VideoThread import VideoThread

# adjustable by user
VIDEO_PATH = "nesquick.mp4"
DEFAULT_ROI = [10, 440]
DEFAULT_ANGLE = 23
# constant
MAIN_FONT = "Helvetica [Cronyx]"
FONT_SIZE = 18
NAME_LIST = ['Tartan', 'Vileda', 'Lacalut', 'Ecodenta',
             'Haus Halt', 'Purina', 'Nesquick', 'Dilmah']


class App(QWidget):
    def createImageWidget(self):
        self.image_parent = QWidget()
        self.image_parent.setMinimumWidth(600)
        self.image_parent.setMinimumHeight(600)
        self.image_parent.installEventFilter(self)
        self.image_parent.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.image_layout = QVBoxLayout()
        self.image_parent.setLayout(self.image_layout)
        self.image_label = QLabel()
        self.image_layout.addWidget(
            self.image_label)
        self.image_label.setStyleSheet(
            "QLabel { background-color : red;}")
        self.image_label.mousePressEvent = self.getPos

    def createItemListAndInputWidgets(self):
        # item list
        self.item_list = QLabel()
        self.item_list.setStyleSheet(
            "QLabel { background-color : white; padding: 5px 10px; }")
        self.item_list.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # aligns text
        self.item_list.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.item_list.setMinimumWidth(300)
        self.item_list.setFont(QFont(MAIN_FONT, FONT_SIZE))
        self.clearList()
        # clear btn
        self.clear_button = QPushButton()
        self.clear_button.setMinimumHeight(120)
        self.clear_button.setText("Clear")
        self.clear_button.setFont(QFont(MAIN_FONT, FONT_SIZE+4))
        self.clear_button.clicked.connect(self.clearList)
        # roi controls
        self.ROI_input_widget = QWidget()
        self.ROI_input_layout = QHBoxLayout(self.ROI_input_widget)
        self.ROI_input_label = QLabel()
        self.ROI_input_label.setText("ROI:")
        self.ROI_input_layout.addWidget(self.ROI_input_label)
        self.ROI_x1 = QLineEdit()
        self.ROI_x1.setValidator(QIntValidator())
        self.ROI_x1.setMaxLength(4)
        self.ROI_x1.setPlaceholderText("x1")
        self.ROI_x1.setText(str(DEFAULT_ROI[0]))
        self.ROI_x1.textEdited.connect(self.changeROI)
        self.ROI_input_layout.addWidget(self.ROI_x1)
        self.ROI_y1 = QLineEdit()
        self.ROI_y1.setValidator(QIntValidator())
        self.ROI_y1.setMaxLength(4)
        self.ROI_y1.setPlaceholderText("y1")
        self.ROI_y1.setText(str(DEFAULT_ROI[1]))
        self.ROI_y1.textEdited.connect(self.changeROI)
        self.ROI_input_layout.addWidget(self.ROI_y1)
        # angle controls
        self.angle_input_widget = QWidget()
        self.angle_input_layout = QHBoxLayout(self.angle_input_widget)
        self.angle_input_label = QLabel("Angle:")
        self.angle_input_layout.addWidget(self.angle_input_label)
        self.angle_input = QLineEdit()
        self.ROI_y1.setPlaceholderText("Degrees")
        self.angle_input.setText(str(DEFAULT_ANGLE))
        self.angle_input.setValidator(QIntValidator())
        self.angle_input.setMaxLength(4)
        self.angle_input.textEdited.connect(self.changeAngle)
        self.angle_input_layout.addWidget(self.angle_input)

    def changeROI(self):
        if self.ROI_x1.text() and self.ROI_x1.text():  # if both fields have values set
            new_ROI = [int(self.ROI_x1.text()), int(self.ROI_y1.text())]
            self.video_thread.updateROI(new_ROI)

    def changeAngle(self):
        try:
            new_angle = int(self.angle_input.text())
            self.video_thread.updateAngle(new_angle)
        except ValueError:
            print("Wrong angle")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveYolo")
        self.cv_img = None
        # on-the-left
        self.createImageWidget()
        # on-the-right
        self.createItemListAndInputWidgets()
        # put everything that's left into layouts
        right_layout_parent = QWidget()
        right_vbox = QVBoxLayout()
        right_layout_parent.setLayout(right_vbox)
        right_layout_parent.setMaximumWidth(400)
        right_vbox.addWidget(self.ROI_input_widget)
        right_vbox.addWidget(self.angle_input_widget)
        right_vbox.addWidget(self.item_list)
        right_vbox.addWidget(self.clear_button)
        main_hbox = QHBoxLayout()
        main_hbox.addStretch(1)
        main_hbox.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        main_hbox.addWidget(self.image_parent)
        main_hbox.addWidget(right_layout_parent, 1)
        main_hbox.addStretch(1)
        self.setLayout(main_hbox)
        # create the video capture thread
        self.video_thread = VideoThread(
            VIDEO_PATH, NAME_LIST, DEFAULT_ROI, DEFAULT_ANGLE)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def scaleAndApplyPixmap(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            self.image_parent.width(), self.image_parent.height(),
            Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setFixedSize(
            scaled_pixmap.rect().width(), scaled_pixmap.rect().height())

    def eventFilter(self, widget, event):
        if (event.type() == QEvent.Resize and
                widget is self.image_parent):
            pixmap = QtGui.QPixmap(self.image_label.pixmap())
            self.scaleAndApplyPixmap(pixmap)
            return True
        return QMainWindow.eventFilter(self, widget, event)

    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        print("x,y:", x, y)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

    def clearList(self):
        self.seen_items = []
        self.item_list.setText("The list is empty")

    @ pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image(self, cv_img, indexes, probs):
        # update pixmap
        qt_img = self.convert_cv_qt(cv_img)
        pixmap = QtGui.QPixmap(qt_img)
        self.scaleAndApplyPixmap(pixmap)
        # update item list
        for i in indexes:
            i = int(i)
            if NAME_LIST[i] not in self.seen_items:
                self.seen_items.append(NAME_LIST[i])
        self.item_list.setText("\n".join(self.seen_items))

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(converted_to_Qt_format)
        return pixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont(MAIN_FONT, FONT_SIZE-2))
    a = App()
    a.show()
    sys.exit(app.exec_())
