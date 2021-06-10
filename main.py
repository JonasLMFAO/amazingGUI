from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2
from PyQt5.QtCore import *
import numpy as np
from VideoThread import VideoThread


# consts
VIDEO_PATH = "nesquick.mp4"
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
        self.image_parent.setStyleSheet(
            "QLabel { background-color : black; padding: 5px; }")
        self.image_layout = QVBoxLayout()
        self.image_parent.setLayout(self.image_layout)
        self.image_label = QLabel()
        self.image_layout.addWidget(
            self.image_label)
        self.image_label.setStyleSheet(
            "QLabel { background-color : red;}")
        self.image_label.mousePressEvent = self.getPos

    def createItemListAndInputWidgets(self):
        self.item_list = QLabel()
        self.item_list.setStyleSheet(
            "QLabel { background-color : white; padding: 5px 10px; }")
        self.item_list.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # aligns text
        self.item_list.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.item_list.setMinimumWidth(300)
        self.item_list.setFont(QFont(MAIN_FONT, FONT_SIZE))
        self.clearList()
        self.clear_button = QPushButton()
        self.clear_button.setMinimumHeight(120)
        self.clear_button.setText("Clear")
        self.clear_button.setFont(QFont(MAIN_FONT, FONT_SIZE+4))
        self.clear_button.clicked.connect(self.clearList)

        self.res_input_widget = QWidget()
        self.res_input_layout = QHBoxLayout(self.res_input_widget)
        self.res_x1 = QLineEdit()
        self.res_x1.setValidator(QIntValidator())
        self.res_x1.setMaxLength(4)
        self.res_x1.setPlaceholderText("x1")
        self.res_input_layout.addWidget(self.res_x1)
        self.res_y1 = QLineEdit()
        self.res_y1.setValidator(QIntValidator())
        self.res_y1.setMaxLength(4)
        self.res_y1.setPlaceholderText("y1")
        self.res_input_layout.addWidget(self.res_y1)

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
        right_vbox.addWidget(self.item_list)
        right_vbox.addWidget(self.res_input_widget)
        right_vbox.addWidget(self.clear_button)
        main_hbox = QHBoxLayout()
        main_hbox.addStretch(1)
        main_hbox.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        main_hbox.addWidget(self.image_parent)
        main_hbox.addWidget(right_layout_parent, 1)
        main_hbox.addStretch(1)
        self.setLayout(main_hbox)

        # create the video capture thread
        self.thread = VideoThread(VIDEO_PATH, NAME_LIST)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def eventFilter(self, widget, event):
        if (event.type() == QEvent.Resize and
                widget is self.image_parent):
            scaled_pixmap = QtGui.QPixmap(self.image_label.pixmap()).scaled(
                self.image_parent.width(), self.image_parent.height(),
                Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setFixedSize(
                scaled_pixmap.rect().width(), scaled_pixmap.rect().height())
            return True
        return QMainWindow.eventFilter(self, widget, event)

    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        print("x,y:", x, y)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def clearList(self):
        self.seen_items = []
        self.item_list.setText("The list is empty")

    @ pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image(self, cv_img, indexes, probs):
        # update pixmap
        qt_img = self.convert_cv_qt(cv_img)
        scaled_pixmap = QtGui.QPixmap(qt_img).scaled(
            self.image_parent.width(), self.image_parent.height(),
            Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setFixedSize(
            scaled_pixmap.rect().width(), scaled_pixmap.rect().height())
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
    a = App()
    a.show()
    sys.exit(app.exec_())
