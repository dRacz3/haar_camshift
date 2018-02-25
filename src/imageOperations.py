import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging


class ImageOperations(object):
    def __init__(self):
        bgSubThreshold = 200
        historyCount = 10
        self.backgroundModel = cv2.createBackgroundSubtractorKNN(historyCount, bgSubThreshold)

        self.trackerWindowName = "Tracker"
        cv2.namedWindow(self.trackerWindowName)
        cv2.createTrackbar("threshold_tolerance", self.trackerWindowName, 150, 255, self.nothing)

    # Basic utility
    def nothing(self, alsonothing):
        pass

    def showIO(self, inputImg, outputImg, name):
        stack = np.hstack((inputImg, outputImg))
        cv2.imshow(name, stack)

    # Returns the resulting image, and the mask
    def removeBackground(self, image, showIO=False):
        # Get mask
        foregroundmask = self.backgroundModel.apply(image)
        # Erode the mask to remove noise in the background
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(foregroundmask, kernel, iterations=1)
        # Dilatation to get back the object
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
        # Apply to original picture
        result = cv2.bitwise_and(image, image, mask=gradient)
        if showIO:
            self.showIO(image, result, "removeBackgroundIO")
        return result, foregroundmask

    def flipImage(self, image):
        return np.fliplr(image)

    def imageThresholding(self, image, showIO=False):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = cv2.getTrackbarPos(
            "threshold_tolerance", self.trackerWindowName)
        ret, thresh = cv2.threshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2)
        result = cv2.bitwise_and(image, image, mask=thresh)
        if showIO:
            cv2.imshow("image thresholding result", thresh)
        return result, thresh

    # Adaptive image threshold based on Gaussian method
    # Returns: result RGB Picture, Mask
    def adaptiveImageThresholding(self, image, showIO=False):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12)
        result = cv2.bitwise_and(image, image, mask=thresh)
        if showIO:
            cv2.imshow("adaptive image thresholding result", thresh)
        return result, thresh

    def getConvexHulls(self, image, mask, showIO=False):
        #        inputimg = cv2.bitwise_not(inputimg)
        img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if showIO:
            drawing = np.zeros(img.shape, np.uint8)
            hull = cv2.convexHull(contours[0])
            cv2.drawContours(drawing, contours, 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, hull, 0, (0, 255, 255), 2)
            cv2.imshow("convex hulls", drawing)

    def removeNoise(self, image, showIO=False):
        pass

    def blurFrame(self, image):
        pass
#        return result
