import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging


class ImageOperations(object):
    def __init__(self):
        bgSubThreshold = 200
        historyCount = 100
        self.backgroundModel = cv2.createBackgroundSubtractorKNN(historyCount, bgSubThreshold )

        self.trackerWindowName = "Tracker"
        cv2.namedWindow(self.trackerWindowName)
        cv2.createTrackbar("threshold_tolerance" ,self.trackerWindowName,150, 255, self.nothing )

    #Basic utility
    def nothing(self, alsonothing):
        pass

    def showIO(self, inputImg, outputImg, name):
        stack = np.hstack((inputImg,outputImg))
        cv2.imshow(name, stack)

    #Returns the resulting image, and the mask
    def removeBackground(self, image, showIO = False):
        #Get mask
        foregroundmask = self.backgroundModel.apply(image)
        #Apply to original picture
        result = cv2.bitwise_and(image, image, mask=foregroundmask)
        if showIO:
            self.showIO(image, result, "removeBackgroundIO")
        return result, foregroundmask


    def imageThresholding(self, image, showIO = False):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = cv2.getTrackbarPos("threshold_tolerance", self.trackerWindowName)
        ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
        result = cv2.bitwise_and(image, image, mask=thresh)
        if showIO:
            cv2.imshow("image thresholding result", result)
        return result, thresh


    def removeNoise(self, image, showIO = False):
        

    def blurFrame(self, image):
        return result