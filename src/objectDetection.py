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
            minSize=(40, 40),
            maxSize=(140, 140),
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
            minSize=(100, 100),
            maxSize=(220, 220),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in results:
            cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 10, (0, 0, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if showIO:
                cv2.imshow("Cascade Head result", img)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,
                            'Head detected :{0}|{1}'.format(int(x + w / 2), int(y + h / 2)),
                            (10,400),
                             font,
                              1,
                              (255,255,255),
                              2,
                              cv2.LINE_AA)
            return (int(x + w /2), int (y+h/2))
        return None



def nothing():
    pass

class colorBasedSegmenter:
    def __init__(self):
        FORMAT = '[%(asctime)-15s][%(levelname)s][%(funcName)s] %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('colorBasedSegmenter')
        self.logger.setLevel('INFO')

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 250, 300)

        # hue  / saturation / value
        cv2.createTrackbar('h_min', 'image', 54, 255, nothing)
        cv2.createTrackbar('s_min', 'image', 64, 255, nothing)
        cv2.createTrackbar('v_min', 'image', 96, 255, nothing)
        cv2.createTrackbar('h_max', 'image', 219, 255, nothing)
        cv2.createTrackbar('s_max', 'image', 255, 255, nothing)
        cv2.createTrackbar('v_max', 'image', 255, 255, nothing)


    def applyColorBasedSegmenetation(self, image, showIO = False):
        hmi = cv2.getTrackbarPos('h_min', 'image')
        smi = cv2.getTrackbarPos('s_min', 'image')
        vmi = cv2.getTrackbarPos('v_min', 'image')
        hma = cv2.getTrackbarPos('h_max', 'image')
        sma = cv2.getTrackbarPos('s_max', 'image')
        vma = cv2.getTrackbarPos('v_max', 'image')
        #put the limits here
        lower_blue = np.array([hmi, smi, vmi])
        upper_blue = np.array([hma, sma, vma])
        print()

        mask = cv2.inRange(image, lower_blue, upper_blue)
            # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image, image, mask=mask)

        try:
            asd = self.findBiggestConvexShapeDeficit(res)
        except:
            pass

        if showIO:
            stack = np.hstack([image, res])
            cv2.imshow('segmentation',stack)
        return res


    #deprecated
    def findBiggestConvexShape(self, image):
        copy = cv2.bitwise_and(image, image)
        gimg = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12)
        im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

        maxArea = 0
        secondMax = 0
        maxContour = contours[0]
        secondContour = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > maxArea:
                secondMax = maxArea
                secondContour = maxContour
                maxArea = area
                maxContour = cnt

        M = cv2.moments(maxContour)
        rect = cv2.minAreaRect(maxContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(copy, [box], 0, (0, 0, 255), 2)

        M = cv2.moments(secondContour)
        rect = cv2.minAreaRect(secondContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(copy, [box], 0, (0, 0, 255), 2)
        cv2.imshow("convexShape", copy)

    def findBiggestConvexShapeDeficit(self, image):
        img = cv2.bitwise_and(image, image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)

        _, contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(img.shape,np.uint8)

        max_area=0

        for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
        cnt=contours[ci]
        hull = cv2.convexHull(cnt)
        prev_hull = cv2.convexHull(cnt)
        prev_cnt = cnt
        moments = cv2.moments(cnt)
        if moments['m00']!=0:
                    cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                    cy = int(moments['m01']/moments['m00']) # cy = M01/M00

        centr=(cx,cy)
        cv2.circle(img,centr,5,[0,0,255],2)
        cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
        cv2.drawContours(drawing,[hull],0,(255,0,255),2)

        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt,returnPoints = False)

        if(1):
                   defects = cv2.convexityDefects(cnt,hull)
                   print(defects[0])
                   mind=0
                   maxd=0
                   for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        dist = cv2.pointPolygonTest(cnt,centr,True)
                        cv2.line(img,start,end,[0,255,0],2)
                        cv2.circle(img,far,5,[0,0,255],-1)
                   i=0
        cv2.imshow('output',drawing)
        cv2.imshow('input',img)
        return cx,cy
