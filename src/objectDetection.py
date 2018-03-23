import logging

import cv2
import numpy as np


class CamShiftTracker(object):
    def __init__(self):
        FORMAT = '[%(asctime)-15s][%(levelname)s][%(funcName)s] %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('camShiftTracker')
        self.logger.setLevel('INFO')

        self.bounding_box = None
        self.track_window = None
        self.w = 150
        self.h = 150

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.gotValidStuffToTrack = False

    def reset(self):
        self.bounding_box = None
        self.track_window = None
        self.w = 150
        self.h = 150

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.gotValidStuffToTrack = False
    def setBoxSize(self, w, h):
        self.w = w
        self.h = h

    def setupFrameAroundValidArea(self, image, bounding_box):
        try:
            self.logger.debug('setupFrameAroundValidArea -> YOU SHOULD SEE IT ONLY ONCE')
            self.bounding_box = bounding_box
            x1 = bounding_box[0]
            x2 = bounding_box[1]
            y1 = bounding_box[2]
            y2 = bounding_box[3]

            self.track_window = (x1, y1, self.w, self.h)
            # set up the ROI for tracking
            roi = image[x1:x2, y1:y2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([40, 60, 32], dtype=np.uint8)
            upper = np.array([180, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower, upper)
            self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
            self.gotValidStuffToTrack = True
            self.logger.debug('Camshift got initialized!')
        except Exception as e:
            self.reset()
            self.logger.debug('Error while initializing: {0} ->Try moving your hand to the center'.format(str(e)))

    def applyCamShift(self, image, bounding_box=None, showIO=False):
        # When we get a new valid initial_location get the data for tracking it later!
        if self.bounding_box is None and bounding_box is not None:
            self.logger.debug('got valid values for hand bounding box, should initialize')
            self.setupFrameAroundValidArea(image, bounding_box)
        # If we have something to track -> Do it
        if self.gotValidStuffToTrack:
            # LOOPED
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

            if showIO:
                # Draw it on image
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)

                # This draws the tracking polygon on the hand
                cv2.polylines(image, [pts], True, 255, 2)
                cv2.imshow('camshift bbox', image)
            if self.checkIfPositionValid(image, ret):
                return ret
            else:
                self.reset()
                return None
        # END LOOP

    def checkIfPositionValid(self, image, ret):
        boundingBoxSize = ret[1]  # contains width | height
        image_shape = image.shape
        self.logger.debug("Bounding size is:{0} ".format(str(boundingBoxSize)))
        # This prevents us from enlarging the bounding shape too much on accident
        shape_factor = 0.8
        if boundingBoxSize[0] > image_shape[0] * shape_factor or boundingBoxSize[1] > image_shape[1] * shape_factor:
            self.logger.info("position is not valid because bounding box size is too big!")
            return False

        # This checks if the box gets too small, then we drop it
        if boundingBoxSize[0] < 30 or boundingBoxSize[1] < 30:
            self.logger.info("position is not valid because bounding box size is too small! Size:[%s|%s]",
                             boundingBoxSize[0], boundingBoxSize[1])
            return False

        # Check if one size is much longer than the other.. it means we are not tracking tha palm
        if boundingBoxSize[0] > boundingBoxSize[1] * 3 or boundingBoxSize[1] > boundingBoxSize[0] * 3:
            self.logger.info("position is not valid because bounding box size is really not symmetrical!")
            return False

        # This checks the position.. if we get no reading -> it is set to around zero
        if int(ret[0][0]) < 5 or int(ret[0][1]) < 5:
            self.logger.info("position is not valid because bounding box center is close to 0,0!")
            return False
        return True


class CascadeClassifierUtils(object):
    def __init__(self):
        #        cascadePath = "haar_finger.xml"
        cascadePath = 'xml/fist.xml'
        facePath = 'xml/frontal_face.xml'
        self.CascadeClassifier = cv2.CascadeClassifier(cascadePath)
        self.FaceCascade = cv2.CascadeClassifier(facePath)

    def evaluateIfHandisFound(self, hand_results):
        if not len(hand_results) == 0:  # Check if we got any result
            for (x, y, w, h) in hand_results:
                print('Hand found at : {0}|{1} Size: {2}|{3}'.format(x,y,w,h))
                return True, (x, y, w, h)
        return False, None

    # Utility function to show the average location of fingers -> good guess for hand position
    def showAverageLocation(self, image, roiframe):
        if roiframe is not None:
            (x1, x2, y1, y2) = self.getBoundingBox(roiframe[0], roiframe[1])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("showAvgLoc", image)

    def getBoundingBox(self, x, y, w , h):
        x1 = int(x - w / 2)
        x2 = int(x + w / 2)
        y1 = int(y - h / 2)
        y2 = int(y + h / 2)

        # normalize
        if x1 < 0:
            x1 = 0
            x2 = x + w
        if y1 < 0:
            y1 = 0
            y2 = y + h

        return x1, x2, y1, y2

    # Returns the results of the haar cascade search, (x,y,w,h) packed to results
    def getHandViaHaarCascade(self, image, showIO=False):
        img = image.copy()
        results = self.CascadeClassifier.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 30),
            maxSize=(150, 220),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in results:
            cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 10, (0, 0, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if showIO:
            cv2.imshow("Cascade result", img)
        return results

    def getFaceViaHaarCascade(self, image, showIO=False):
        img = image.copy()
        results = self.FaceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(50, 60),
            maxSize=(300, 400),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in results:
            cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 10, (0, 0, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if showIO:
            cv2.imshow("Cascade Head result", img)
        return results
