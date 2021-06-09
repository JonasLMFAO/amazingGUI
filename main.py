from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2
from PyQt5.QtCore import *
import numpy as np
import torch
from LiveModel_returns_boxes import LiveYolo
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from threading import Thread
from time import sleep

# model init
name_list = ["Coca Cola", "Lacalut", "Persil", "Paper Clips", "Colgate"]
live_model = LiveYolo()
live_model.load()

# consts
MAIN_FONT = QFont("Helvetica [Cronyx]", 16)
ROI = ((10, 440), (1260, 1340))  # ((440, 10), (1340, 1260))


def drawBoxes(frame, pred):
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{name_list[c]} {conf:.2f}'
                p = plot_one_box(xyxy, frame, label=label,
                                 color=colors(c, True), line_thickness=2)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture("new.mp4")
        cv_img = None
        self.pred = None

        def getPred():
            while True:
                if cv_img is not None:
                    self.pred = live_model.run_on_single_frame(
                        cv_img[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]])
        thread = Thread(target=getPred)
        thread.start()
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                with torch.no_grad():
                    if self.pred is not None:
                        drawBoxes(cv_img, self.pred)
                        indexes = np.asarray(self.pred[0][:, -1])
                        probs = np.asarray(self.pred[0][:, -2])
                        self.change_pixmap_signal.emit(cv_img, indexes, probs)
            sleep(0.08)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveYolo")

        # Widgets
        self.image_label = QLabel(self)
        self.image_label.setMinimumWidth(600)
        self.image_label.setMinimumHeight(600)
        self.image_label.installEventFilter(self)
        self.image_label.setStyleSheet(
            "QLabel { background-color : white; padding: 5px; }")
        self.image_label.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.item_list = QLabel(self)
        self.item_list.setStyleSheet(
            "QLabel { background-color : white; padding: 5px 10px; }")
        self.item_list.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.item_list.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.item_list.setMinimumWidth(300)
        self.item_list.setFont(MAIN_FONT)

        self.clearList()
        self.clear_button = QPushButton(self)
        self.clear_button.setMinimumHeight(100)
        self.clear_button.setText("Clear")
        self.clear_button.setFont(MAIN_FONT)
        self.clear_button.clicked.connect(self.clearList)

        # Layouts
        v_layout = QVBoxLayout()
        v_widget = QWidget()
        v_widget.setLayout(v_layout)
        v_widget.setMaximumWidth(400)
        v_layout.addWidget(self.item_list)
        v_layout.addWidget(self.clear_button)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        hbox.addWidget(self.image_label)
        hbox.addWidget(v_widget, 1)
        hbox.addStretch(1)
        self.setLayout(hbox)

        self.cv_img = None

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def eventFilter(self, widget, event):
        if (event.type() == QEvent.Resize and
                widget is self.image_label):
            self.image_label.setPixmap(QtGui.QPixmap(self.image_label.pixmap()).scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio))
            return True
        return QMainWindow.eventFilter(self, widget, event)

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
        self.image_label.setPixmap(qt_img.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio))
        # update item list
        for i in indexes:
            i = int(i)
            if name_list[i] not in self.seen_items:
                self.seen_items.append(name_list[i])
        self.item_list.setText("\n".join(self.seen_items))

    def drawROI(self, cv_img):
        color = (0, 255, 0)
        thickness = 5
        cv_img = cv2.rectangle(cv_img, ROI[0], ROI[1], color, thickness)

    def convert_cv_qt(self, cv_img):
        self.drawROI(cv_img)
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
