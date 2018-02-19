import cv2
import numpy as np
import time


class background_remover():
    def __init__(self, binaryTreshold = 60, subTreshold = 700):
        #Background subtraction
        self.threshold = binaryTreshold  #  BINARY threshold
        self.bgSubThreshold = subTreshold
        self.bgModel = cv.createBackgroundSubtractorKNN(0, self.bgSubThreshold)

    #This function removes the background from a given frame via the KNN algorithm
    def remove_bg(self, frame):
        fgmask = self.bgModel.apply(frame)
        frame_without_bg = cv.bitwise_and(frame, frame, mask=fgmask)
        return frame_without_bg
