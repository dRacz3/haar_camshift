import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging


class ImageOperations(object):
    def __init__(self):
        bgSubThreshold = 100
        historyCount = 25
        self.backgroundModel = cv2.createBackgroundSubtractorKNN(historyCount, bgSubThreshold)

        cascadePath = "haar_finger.xml"
        self.CascadeClassifier = cv2.CascadeClassifier(cascadePath)
        faceCascadePath = 'haarcascade_frontalface_alt.xml'
        self.FaceCascadeClassifier = cv2.CascadeClassifier(faceCascadePath)

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
        self.initial_location = None

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
        # Apply gaussian filter to smoothen , then median to remove more noise from mask
        gaussian = cv2.GaussianBlur(foregroundmask, (1, 1), 0)
        # Erode the mask to remove noise in the background
        erosion_kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(gaussian, erosion_kernel, iterations=1)
        # Dilatation to get back the object
        dilation_kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(erosion, dilation_kernel, iterations=1)
        # Apply to original picture
        result = cv2.bitwise_and(image, image, mask=erosion)
        if showIO:
            self.showIO(image, result, "removeBackgroundIO")
            # self.showIO(gaussian, median, 'Gaussian-Median filter effect on background')
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

    # Returns the results of the haar cascade search, (x,y,w,h) packed to results
    def getHandViaHaarCascade(self, image, showIO=False):
        img = image.copy()
        results = self.CascadeClassifier.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=15,
            minSize=(20, 30),
            maxSize=(50, 120),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in results:
            #        pyautogui.moveTo(sizeX-x*3,y*3)
            cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 10, (0, 0, 255), -1)
    #        cv2.circle(frame, (x+w/2, y+h/2))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if showIO:
            cv2.imshow("Cascade result", img)
        return results

    def getFaceViaHaarCascade(self, image, showIO=False):
        img = image.copy()
        results = self.FaceCascadeClassifier.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=15,
            minSize=(20, 30),
            maxSize=(50, 120),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)

        for (x, y, w, h) in results:
            #        pyautogui.moveTo(sizeX-x*3,y*3)
            cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 10, (255, 0, 255), -1)
    #        cv2.circle(frame, (x+w/2, y+h/2))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        if showIO:
            cv2.imshow("Face Cascade result", img)

    def applyCamShift(self, image, initial_location, showIO=False):
        # setup initial location of window
        if self.initial_location is None:
            self.initial_location = initial_location

        for (r, h, c, w) in self.initial_location:
            self.track_window = (c, r, w, h)

        # set up the ROI for tracking
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, self.track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imshow('Camshift', img2)
##

    def evaluateIfHandisFound(self, fingers_results):
        if not len(fingers_results) == 0:  # Check if we got any result
            # dumb check for finger count
            foundFingers = 0
            for (x, y, w, h) in fingers_results:
                foundFingers = foundFingers + 1
            # if we have more than 3 match, take avg, and say we found a hand
            if foundFingers > 3:
                return True, self.calcAverageLocation(fingers_results)
        # If nothing valid is found say we didnt find it, and return none as position
        return False, None

    def calcAverageLocation(self, location_frames):
        x_avg = 0
        y_avg = 0
        count = 0
        for (x, y, w, h) in location_frames:
            x_avg = x_avg + ((x + w) / 2)
            y_avg = y_avg + ((y + h) / 2)
            count = count + 1
        x_avg = x_avg / count
        y_avg = y_avg / count
        return x, y

    def getConvexHulls(self, image, mask, showIO=False):
        #        inputimg = cv2.bitwise_not(inputimg) #negate image
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
