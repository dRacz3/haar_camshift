import logging

import cv2
import numpy as np
from objectDetection import CamShiftTracker, CascadeClassifierUtils


class ImageOperations(object):
    def __init__(self):
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('imageOperations')
        self.logger.setLevel('DEBUG')
        bgSubThreshold = 80
        historyCount = 40
        self.backgroundModel = cv2.createBackgroundSubtractorKNN(historyCount, bgSubThreshold)
        self.CascadeClassifierUtils = CascadeClassifierUtils()
        self.initial_location = None
        self.camShiftTracker = CamShiftTracker()
        self.logger.info("Image operations loaded and initialized!")

    def showIO(self, inputImg, outputImg, name):
        stack = np.hstack((inputImg, outputImg))
        cv2.imshow(name, stack)

    # Returns the resulting image, and the mask
    def removeBackground(self, image, showIO=False, debug=False):
        # Get mask
        fakemask = cv2.GaussianBlur(image, (15, 15), 0)
        #nonprocessedMask = self.backgroundModel.apply(image)
        foregroundmask = self.backgroundModel.apply(fakemask)
        # Apply gaussian filter to smoothen
        gaussian = cv2.GaussianBlur(foregroundmask, (5, 5), 0)
        # Erode the mask to remove noise in the background
        erosion_kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(gaussian, erosion_kernel, iterations=1)
        # Dilatation to get back the object
        dilation_kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(erosion, dilation_kernel, iterations=1)
        # Apply to original picture
        result = cv2.bitwise_and(image, image, mask=dilation)
        #result = image
        if debug:
            cv2.imshow('Gaussian_Preprocessed', nonprocessedMask)
            cv2.imshow('foregroundmask', foregroundmask)
            cv2.imshow('Gaussian Mask', gaussian)
            cv2.imshow('Erosio', erosion)
            cv2.imshow('Dilatation', dilation)
        if showIO:
            self.showIO(image, result, "removeBackgroundIO")
            # self.showIO(gaussian, median, 'Gaussian-Median filter effect on background')
        return result, foregroundmask

    def flipImage(self, image):
        return np.fliplr(image)

    # Adaptive image threshold based on Gaussian method
    # Returns: result RGB Picture, Mask
    def adaptiveImageThresholding(self, image, showIO=False):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12)
        result = cv2.bitwise_and(image, image, mask=thresh)
        if showIO:
            cv2.imshow("adaptive image thresholding result", result)
        return result, thresh

    # This algorithm applies the camShift algorithm if proper initial_location is given
    # Or it has already been initialized and not lost track of the tracked object
    def applyCamShift(self, image, initial_location=None, showIO=False):
        # If the initial location passed is not none -> we have to initialize
        result = None
        if initial_location is not None:
            # Get the ROI frame box, pass it on
            self.logger.debug('Initial location has been passed, should initialize')
            boundingBox = self.CascadeClassifierUtils.getBoundingBox(initial_location[0], initial_location[1])
            result = self.camShiftTracker.applyCamShift(image=image, bounding_box=boundingBox, showIO=showIO)
        else:
            # normal call should be this, when we are already initialized
            result = self.camShiftTracker.applyCamShift(image=image, showIO=showIO)
        # it means that we got a valid result!
        return result

    def getHandViaHaarCascade(self, image, showIO=False):
        return self.CascadeClassifierUtils.getHandViaHaarCascade(image, showIO)

    # Utility to reset the camshift tracker
    def resetCamShift(self):
        self.camShiftTracker.reset()

    def calculate_manhattan_distance(self, haar_result, camshift_result):
        hx = haar_result[0][0]
        hy = haar_result[0][1]

        cx = camshift_result[0][0]
        cy = camshift_result[0][1]

        man_dist = abs(hx - cy) + abs(hy - cy)
        return man_dist
