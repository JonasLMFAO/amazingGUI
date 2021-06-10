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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveYolo")

        # Widgets
        self.image_label = QWidget(self)
        self.image_label.setMinimumWidth(600)
        self.image_label.setMinimumHeight(600)
        self.image_label.installEventFilter(self)
        self.image_label.setStyleSheet(
            "QLabel { background-color : white; padding: 5px; }")

        self.image_sublayout = QVBoxLayout()
        self.image_label.setLayout(self.image_sublayout)
        self.image_sublabel = QLabel(self)
        self.image_sublabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_sublayout.addWidget(self.image_sublabel)
        self.image_sublabel.setStyleSheet(
            "QLabel { background-color : red; padding: 5px; }")
        self.image_sublabel.mousePressEvent = self.getPos

        self.item_list = QLabel(self)
        self.item_list.setStyleSheet(
            "QLabel { background-color : white; padding: 5px 10px; }")
        self.item_list.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.item_list.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.item_list.setMinimumWidth(300)
        self.item_list.setFont(QFont(MAIN_FONT, FONT_SIZE))

        self.clearList()
        self.clear_button = QPushButton(self)
        self.clear_button.setMinimumHeight(120)
        self.clear_button.setText("Clear")
        self.clear_button.setFont(QFont(MAIN_FONT, FONT_SIZE+4))
        self.clear_button.clicked.connect(self.clearList)

        # Layouts
        v_layout = QVBoxLayout()
        v_widget = QWidget()
        v_widget.setLayout(v_layout)
        v_widget.setMaximumWidth(400)
        v_layout.addWidget(self.item_list)
        v_layout.addWidget(self.clear_button)
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        h_layout.addWidget(self.image_label)
        h_layout.addWidget(v_widget, 1)
        h_layout.addStretch(1)
        self.setLayout(h_layout)

        self.cv_img = None

        # create the video capture thread
        self.thread = VideoThread(VIDEO_PATH, NAME_LIST)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def eventFilter(self, widget, event):
        if (event.type() == QEvent.Resize and
                widget is self.image_label):
            scaled_pixmap = QtGui.QPixmap(self.image_sublabel.pixmap()).scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio)
            self.image_sublabel.setPixmap(scaled_pixmap)
            self.image_sublabel.setFixedSize(
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

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image(self, cv_img, indexes, probs):
        # update pixmap
        qt_img = self.convert_cv_qt(cv_img)
        self.image_sublabel.setPixmap(qt_img.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio))
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
