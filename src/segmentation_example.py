import numpy as np
import cv2

import logging

from image_proc_utils import mouseMotionManager


class CameraMouse(object):
    def __init__(self):
        self.hsv_tuning = 'Tuner'
        self.th_window_name = 'Threshold_tuner'
        self.hmin = 0
        self.smin = 171
        self.vmin = 200
        self.hmax = 194
        self.smax = 255
        self.vmax = 251

        self.th_min = 0
        self.th_max = 255

        self.tunersAreCreated = False

        bgSubThreshold = 250
        self.backgroundModel = cv2.createBackgroundSubtractorKNN(1000, bgSubThreshold)

        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        self.mouseManager = mouseMotionManager(frame)

    def nothing(self, x):
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

        cv2.namedWindow(self.th_window_name)
        cv2.createTrackbar('Th_min', self.th_window_name, self.th_min, 255, self.nothing)
        cv2.createTrackbar('Th_max', self.th_window_name, self.th_max, 255, self.nothing)

    def color_treshold(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        if self.tunersAreCreated:
            self.hmin = cv2.getTrackbarPos('H_MIN', self.hsv_tuning)
            self.smin = cv2.getTrackbarPos('S_MIN', self.hsv_tuning)
            self.vmin = cv2.getTrackbarPos('V_MIN', self.hsv_tuning)
            self.hmax = cv2.getTrackbarPos('H_MAX', self.hsv_tuning)
            self.smax = cv2.getTrackbarPos('S_MAX', self.hsv_tuning)
            self.vmax = cv2.getTrackbarPos('V_MAX', self.hsv_tuning)

        lower = np.array([self.hmin, self.smin, self.vmin])
        upper = np.array([self.hmax, self.smax, self.vmax])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)
        return res

    def gray_treshold(self, img):
        if self.tunersAreCreated:
            self.th_min = cv2.getTrackbarPos('Th_min', self.th_window_name)
            self.th_max = cv2.getTrackbarPos('Th_max', self.th_window_name)
        ret, thresh = cv2.threshold(img, self.th_min, self.th_max, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def removeBackground(self, image, showIO=False):
        # Removing background as much as possible
        fgmask = self.backgroundModel.apply(image)
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 1, 1)
        #cv2.imshow('No BG', fgmask)
        res = cv2.bitwise_and(image, image, mask=fgmask)
        if showIO:
            stack = np.hstack((image, res))
            cv2.imshow('removeBackground', stack)
        return res

    # Apply thresholding to leave MOSTLY skin
    def applySegmentationBasedonHSV(self, img, showIO=False):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Blurring it a bit
        blurred = cv2.GaussianBlur(hsv, (7, 7), 1, 1)
        blurred = self.color_treshold(hsv)
        if showIO:
            stack = np.hstack((img, blurred))
            cv2.imshow('applySegmentationBasedonHSV', stack)
        return blurred

    def applyConvexHull(self, inputimg, originalImg=None, showIO=False):
        # COUNTOUR DETECTION
        frame = originalImg
        inputimg = cv2.bitwise_not(inputimg)

        img, contours, hierarchy = cv2.findContours(inputimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('contours', img)
        drawing = np.zeros(img.shape, np.uint8)
        max_area = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                ci = i
        cnt = contours[ci]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

        centr = (cx, cy)
        cv2.circle(img, centr, 5, [0, 0, 255], 2)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

        cv2.circle(frame, centr, 5, [0, 0, 255], 2)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)

        defects = cv2.convexityDefects(cnt, hull)
        mind = 0
        maxd = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dist = cv2.pointPolygonTest(cnt, centr, True)
            cv2.line(img, start, end, [0, 255, 0], 2)
            cv2.line(frame, start, end, [0, 255, 0], 2)

            cv2.circle(img, far, 5, [0, 0, 255], -1)
            cv2.circle(frame, far, 5, [0, 0, 255], -1)
        print(i)

        if showIO:
            cv2.imshow('convexHUll', frame)

            hsv_tuning = 'Tuner'
            th_window_name = 'Threshold_tuner'

        return cx, cy, i

    def runProgram(self):
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            frame_no_bg = self.removeBackground(frame, showIO=True)
        #    edges = cv2.Canny(frame_no_bg,100,100)
        #    cv2.imshow('edges', edges)

            # Converting to HUE-SATURATION image
            segmentedImg = self.applySegmentationBasedonHSV(frame_no_bg, showIO=True)
            normal = cv2.cvtColor(segmentedImg, cv2.COLOR_HSV2RGB)
            hsv_rgb_stack = np.hstack((segmentedImg, normal))
            #cv2.imshow("hsv-> rgb", hsv_rgb_stack)

            # Making it greyscale
            gray = cv2.cvtColor(frame_no_bg, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (1, 1), 1, 1)
            binary_thresholded = self.gray_treshold(gray)

            gray_binary_stack = np.hstack((gray, binary_thresholded))
            #cv2.imshow("gray_binary_stack", gray_binary_stack)

            kernel_close = np.ones((1, 1), np.uint8)
            kernel = np.ones((1, 1), np.uint8)
            kernel_erode = np.ones((5, 5), np.uint8)
            erode = cv2.erode(binary_thresholded, kernel_erode, iterations=3)
            closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)

            try:
                cx, cy, points = self.applyConvexHull(closing, originalImg=frame, showIO=True)
            except Exception as e:
                print(e)
                continue
            self.mouseManager.move(cx, cy, points)
            # mouseManager.release()
            # exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.cap.release()
        self.mouseManager.release()
        cv2.destroyAllWindows()
###########################x MAIN CODE x#####################################


app = CameraMouse()
app.create_tuner()
app.runProgram()
