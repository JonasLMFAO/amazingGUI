from os import stat
from threading import Thread, Lock
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
ROI_OFFSET = (1250, 700)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, VIDEO_PATH, NAME_LIST, DEFAULT_ROI, DEFAULT_ANGLE):
        super().__init__()
        self._run_flag = True
        self.VIDEO_PATH = VIDEO_PATH
        self.NAME_LIST = NAME_LIST
        self.updateROI(DEFAULT_ROI)
        self.IMG_ANGLE = DEFAULT_ANGLE

    def updateROI(self, new_ROI):
        self.ROI = ((new_ROI[0], new_ROI[1]),
                    (new_ROI[0] + ROI_OFFSET[0], new_ROI[1]+ROI_OFFSET[1]))

    def drawBoxes(self, frame, pred):
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{self.NAME_LIST[c]} {conf:.2f}'
                    # adjust box coords based on the self.ROI
                    adjusted_xyxy = [xyxy[0] + self.ROI[0][0], xyxy[1] +
                                     self.ROI[0][1], xyxy[2] + self.ROI[0][0], xyxy[3] + self.ROI[0][1]]
                    p = plot_one_box(adjusted_xyxy, frame, label=label,
                                     color=colors(c, True), line_thickness=3)

    def updatePredictionLoop(self):
        while self._run_flag:
            if self.cv_img is not None:
                # protect self.cv_img from being changed on an another thread
                self.threading_lock.acquire()
                cut_out_ROI = np.copy(self.cv_img[self.ROI[0][1]:self.ROI[1]
                                                  [1], self.ROI[0][0]:self.ROI[1][0]])
                self.threading_lock.release()
                self.pred = live_model.run_on_single_frame(cut_out_ROI)

    def drawROI(self, cv_img):
        color = (0, 255, 0)
        thickness = 6
        cv_img = cv2.rectangle(
            cv_img, tuple(self.ROI[0]), tuple(self.ROI[1]), color, thickness)

    def rotatedImage(self, image):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(
            image_center, self.IMG_ANGLE, 1.0)
        image = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return image

    def updateAngle(self, new_angle):
        self.IMG_ANGLE = new_angle

    def run(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        self.cv_img = None
        self.pred = None
        self.threading_lock = Lock()
        self.model_thread = Thread(target=self.updatePredictionLoop)
        self.model_thread.start()
        while self._run_flag:
            ret, cv_img = cap.read()
            cv_img = self.rotatedImage(cv_img)
            self.drawROI(cv_img)
            self.cv_img = cv_img
            if ret:
                with torch.no_grad():
                    if self.pred is not None:
                        # here we use thread lock because self.pred is on an another thread
                        self.threading_lock.acquire()
                        self.drawBoxes(cv_img, self.pred)
                        indexes = np.asarray(self.pred[0][:, -1])
                        probs = np.asarray(self.pred[0][:, -2])
                        self.threading_lock.release()
                        self.change_pixmap_signal.emit(
                            cv_img, indexes, probs)

            sleep(0.08)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.model_thread.join()
        self.wait()
