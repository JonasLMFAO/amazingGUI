from threading import Thread
from time import sleep
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread
import torch
from LiveModel_returns_boxes import LiveYolo
from utils.plots import colors, plot_one_box
import cv2


# model init

live_model = LiveYolo()
live_model.load()


ROI = ((10, 440), (1260, 1140))  # ((440, 10), (1340, 1260))


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, video_path, name_list):
        super().__init__()
        self._run_flag = True
        self.VIDEO_PATH = video_path
        self.NAME_LIST = name_list

    def drawBoxes(self, frame, pred):
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{self.NAME_LIST[c]} {conf:.2f}'
                    # adjust box coords based on the ROI
                    adjusted_xyxy = [xyxy[0] + ROI[0][0], xyxy[1] +
                                     ROI[0][1], xyxy[2] + ROI[0][0], xyxy[3] + ROI[0][1]]
                    p = plot_one_box(adjusted_xyxy, frame, label=label,
                                     color=colors(c, True), line_thickness=3)

    def updatePredictionLoop(self):
        while True:
            if self.cv_img is not None:
                cropped = self.cv_img[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]]
                self.pred = live_model.run_on_single_frame(cropped)

    def drawROI(self, cv_img):
        color = (0, 255, 0)
        thickness = 6
        cv_img = cv2.rectangle(cv_img, ROI[0], ROI[1], color, thickness)

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def run(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.cv.CV_CAP_PROP_FPS,15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        self.cv_img = None
        self.pred = None
        thread = Thread(target=self.updatePredictionLoop)
        thread.start()
        while self._run_flag:
            ret, self.cv_img = cap.read()
            if ret:
                with torch.no_grad():
                    if self.pred is not None:
                        self.drawBoxes(self.cv_img, self.pred)
                        self.drawROI(self.cv_img)
                        indexes = np.asarray(self.pred[0][:, -1])
                        probs = np.asarray(self.pred[0][:, -2])
                        self.change_pixmap_signal.emit(
                            self.cv_img, indexes, probs)
            sleep(0.08)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
