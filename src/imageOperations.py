import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging


class ImageOperations(object):
    def __init__(self):
        bgSubThreshold = 150
        historyCount = 5
        self.backgroundModel = cv2.createBackgroundSubtractorKNN(historyCount, bgSubThreshold)

        self.hmin = 0
        self.smin = 171
        self.vmin = 200
        self.hmax = 194
        self.smax = 255
        self.vmax = 251

        self.tunersAreCreated = False
        self.trackerWindowName = "Tracker"

        self.hsv_tuning = "HSV Tuner"

        cv2.namedWindow(self.trackerWindowName)
        cv2.createTrackbar("threshold_tolerance", self.trackerWindowName, 150, 255, self.nothing)

        self.create_tuner()

    # Basic utility
    def nothing(self, alsonothing):
        pass

    def create_tuner(self):
        # create trackbars for color change
        self.tunersAreCreated = True
        cv2.namedWindow(self.hsv_tuning)
        cv2.createTrackbar('H_MIN', self.hsv_tuning, self.hmin, 255, self.nothing)
        cv2.createTrackbar('S_MIN', self.hsv_tuning, self.smin, 255, self.nothing)
        cv2.createTrackbar('V_MIN', self.hsv_tuning, self.vmin, 255, self.nothing)
        cv2.createTrackbar('H_MAX', self.hsv_tuning, self.hmax, 255, self.nothing)
        cv2.createTrackbar('S_MAX', self.hsv_tuning, self.smax, 255, self.nothing)
        cv2.createTrackbar('V_MAX', self.hsv_tuning, self.vmax, 255, self.nothing)

    def showIO(self, inputImg, outputImg, name):
        stack = np.hstack((inputImg, outputImg))
        cv2.imshow(name, stack)

    # Returns the resulting image, and the mask
    def removeBackground(self, image, showIO=False):
        # Get mask
        foregroundmask = self.backgroundModel.apply(image)
        # Erode the mask to remove noise in the background
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(foregroundmask, kernel, iterations=1)
        # Dilatation to get back the object
        dilation = cv2.dilate(erosion, kernel, iterations=10)
        #gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
        # Apply to original picture
        result = cv2.bitwise_and(image, image, mask=dilation)
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
            cv2.imshow("image thresholding result", result)
        return result, thresh

    # Adaptive image threshold based on Gaussian method
    # Returns: result RGB Picture, Mask
    def adaptiveImageThresholding(self, image, showIO=False):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12)
        result = cv2.bitwise_and(image, image, mask=thresh)
        if showIO:
            cv2.imshow("adaptive image thresholding result", result)
        return result, thresh

    def color_treshold(self, img, showIO=False):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if self.tunersAreCreated:
            self.hmin = cv2.getTrackbarPos('H_MIN', self.hsv_tuning)
            self.smin = cv2.getTrackbarPos('S_MIN', self.hsv_tuning)
            self.vmin = cv2.getTrackbarPos('V_MIN', self.hsv_tuning)
            self.hmax = cv2.getTrackbarPos('H_MAX', self.hsv_tuning)
            self.smax = cv2.getTrackbarPos('S_MAX', self.hsv_tuning)
            self.vmax = cv2.getTrackbarPos('V_MAX', self.hsv_tuning)

        # define range of desired color
        lower = np.array([self.hmin, self.smin, self.vmin])
        upper = np.array([self.hmax, self.smax, self.vmax])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)
        if showIO:
            self.showIO(img, res, "color threshold")
        return res

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
