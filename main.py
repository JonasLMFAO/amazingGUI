from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import torch
from LiveYolo import LiveYolo

name_list = ["null", "Coca Cola", "Lacalut", "Persil", "Paper Clips", "Colgate"]
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
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image

        self.image_label = QLabel(self)
        #self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.item_list = QTextEdit(self)
        self.item_list.setReadOnly(True)
        self.item_list.setMinimumWidth(200)
        #self.item_list.setLineWrapMode(QTextEdit.NoWrap)
        self.clear_button = QPushButton(self)
        self.clear_button.setText("Clear")

        vbox = QVBoxLayout()
        vbox.addWidget(self.item_list, 1)
        vbox.addWidget(self.clear_button)

        self.start_coords = (0, 0)
        self.end_coords = (50, 50)
        self.cv_img = None

        # create a vertical box layout and add the two labels
        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label, 1)
        hbox.addLayout(vbox, 1)
        # set the hbox layout as the widgets layout
        self.setLayout(hbox)

        self.frame_number = 0
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()



    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image(self, cv_img, indexes, probs):
        """Updates the image_label with a new opencv image"""
        print(indexes)
        qt_img = self.convert_cv_qt(cv_img)
        self.item_list.insertPlainText(f"[{self.frame_number}]{[name_list[int(i)] for i in indexes][:5]}\n")
        self.frame_number += 1
        sb = self.item_list.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv_img #cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
