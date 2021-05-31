from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import torch
from time import time
from LiveYolo import LiveYolo

name_list = ["Coca Cola", "Lacalut", "Persil", "Paper Clips", "Colgate"]
live_model = LiveYolo()
live_model.load()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture("rotated.mp4")
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                start_time = time()
                with torch.no_grad():
                    cv_img, indexes, probs = live_model.run_on_single_frame(cv_img)
                    indexes = indexes.numpy()
                    probs = probs.numpy()
                print(time()-start_time)
                self.change_pixmap_signal.emit(cv_img, indexes, probs)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        # create the label that holds the image

        self.image_label = QLabel(self)
        self.img_width = 640
        self.img_height = 480
        # create a text label
        self.item_list = QLabel(self)
        self.item_list.setStyleSheet("QLabel { background-color : white; padding: 5px 10px}");
        self.item_list.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.item_list.setMinimumWidth(200)
        self.clearList()
        self.clear_button = QPushButton(self)
        self.clear_button.setText("Clear")
        self.clear_button.clicked.connect(self.clearList)

        vbox = QVBoxLayout()
        vbox.addWidget(self.item_list, 1)
        vbox.addWidget(self.clear_button)

        self.start_coords = (0, 0)
        self.end_coords = (50, 50)
        self.cv_img = None

        # create a vertical box layout and add the two labels
        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        hbox.addLayout(vbox, 1)
        # set the hbox layout as the widgets layout
        self.setLayout(hbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def clearList(self):
        self.seen_items = []
        self.item_list.setText("The list is empty")

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image(self, cv_img, indexes, probs):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        for i in indexes:
            i = int(i)
            if name_list[i] not in self.seen_items:
                self.seen_items.append(name_list[i])
        self.item_list.setText("\n".join(self.seen_items))
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(converted_to_Qt_format)
        pixmap_scaled = pixmap.scaled(self.img_width, self.img_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pixmap_scaled


if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
