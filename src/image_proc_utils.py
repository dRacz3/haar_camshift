import cv2 as cv
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
        fgmask = cv.medianBlur(fgmask, 15)
        cv.imshow('fgmask', fgmask)

        frame_without_bg = cv.bitwise_and(frame, frame, mask=fgmask)
        return frame_without_bg

class camshift_tracker():
    def __init__(self, initial_frame):
        # setup initial location of window
        r,h,c,w = 250,90,125,125  # simply hardcoded the values
        self.track_window = (c,r,w,h)
        # set up the ROI for tracking # -> bounding box
        roi = initial_frame[r:r+h, c:c+w]
        hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv.normalize(self.roi_hist,self.roi_hist,0,255,cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 5, 10 )

    def process(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        # apply meanshift to get the new location
        self.ret, self.track_window = cv.CamShift(dst, self.track_window, self.term_crit)
        drawing = self.draw_on_img(frame)
        return drawing

    def draw_on_img(self,frame):
        # Draw it on image
        pts = cv.boxPoints(self.ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        return img2

class haar_classifier():
    def __init__(self, cascPath):
        self.Cascade = cv.CascadeClassifier(cascPath)

    def process(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        results = self.Cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=26,
            minSize=(10, 40),
            maxSize = (120, 300),
            flags=cv.CASCADE_FIND_BIGGEST_OBJECT
        )
        cv.imshow('gray_haar', self.draw_on_img(gray, results))
        return results

    def draw_on_img(self,frame, results):
        for (x, y, w, h) in results:
            cv.circle(frame,(int(x + w/2) ,int(y + h/2)), 10, (0,0,255), -1)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame
