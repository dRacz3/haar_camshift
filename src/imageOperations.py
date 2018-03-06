import cv2
import numpy as np
import logging


class camShiftTracker(object):
    def __init__(self):
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

    def setupFrameAroundValidArea(self, image, bounding_box):
        try:
            print('setupFrameAroundValidArea -> YOU SHOULD SEE IT ONLY ONCE')
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
            print('Camshift got initialized!')
        except Exception as e:
            self.reset()
            print('Error while initializing: {0} ->Try moving your hand to the center'.format(e))

    def applyCamShift(self, image, bounding_box=None, showIO=False):
        # When we get a new valid initial_location get the data for tracking it later!
        if self.bounding_box is None and bounding_box is not None:
            print('got valid values for hand bounding box, should initialize')
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
                img2 = cv2.polylines(image, [pts], True, 255, 2)

            if self.checkIfPositionValid(image, ret):
                return ret
            else:
                self.reset()
                return None
        # END LOOP

    def checkIfPositionValid(self, image, ret):
        boundingBoxSize = ret[1]  # contains width | height
        imageShape = image.shape
        print("Bounding size is: ", boundingBoxSize)
        # This prevents us from enlarging the bounding shape too much on accident
        shapeFactor = 0.4
        if(boundingBoxSize[0] > imageShape[0] * shapeFactor or boundingBoxSize[1] > imageShape[1] * shapeFactor):
            return False

        # This checks if the box gets too small, then we drop it
        if(boundingBoxSize[0] < 70 and boundingBoxSize[1] < 70):
            return False

        # Check if one size is much longer than the other.. it means we are not tracking tha palm
        if(boundingBoxSize[0] > boundingBoxSize[1] * 3 or boundingBoxSize[1] > boundingBoxSize[0] * 3):
            return False

        # This checks the position.. if we get no reading -> it is set to around zero
        if(int(ret[0][0]) < 5 or int(ret[0][1]) < 5):
            return False
        return True


class CascadeClassifierUtils(object):
    def __init__(self):
        cascadePath = "haar_finger.xml"
        self.CascadeClassifier = cv2.CascadeClassifier(cascadePath)

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

    # This function calculates the average location of all detected fingers
    # TODO : Improve this by counting only local fingers as one, and neglect the others
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

    # Utility function to show the average location of fingers -> good guess for hand position
    def showAverageLocation(self, image, roiframe):
        if roiframe is not None:
            (x1, x2, y1, y2) = self.getBoundingBox(roiframe[0], roiframe[1])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("showAvgLoc", image)

    # Quick calculation to get a given shaped box for center coordiantes
    # WARNING : Hard coded width + height!
    def getBoundingBox(self, x, y):
        x = x
        y = y
        w = 150
        h = 150
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
            minNeighbors=5,
            minSize=(20, 30),
            maxSize=(50, 120),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in results:
            cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 10, (0, 0, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if showIO:
            cv2.imshow("Cascade result", img)
        return results


class ImageOperations(object):
    def __init__(self):
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('imageOperations')
        self.logger.setLevel('DEBUG')
        bgSubThreshold = 100
        historyCount = 2
        self.backgroundModel = cv2.createBackgroundSubtractorKNN(historyCount, bgSubThreshold)
        self.CascadeClassifierUtils = CascadeClassifierUtils()
        self.initial_location = None
        self.camShiftTracker = camShiftTracker()
        self.logger.info("Image operations loaded and initialized!")

    def showIO(self, inputImg, outputImg, name):
        stack = np.hstack((inputImg, outputImg))
        cv2.imshow(name, stack)

    # Returns the resulting image, and the mask
    def removeBackground(self, image, showIO=False):
        # Get mask
        fakemask = cv2.GaussianBlur(image, (5, 5), 0)
        foregroundmask = self.backgroundModel.apply(fakemask)
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
            print('Initial location has been passed, should initialize')
            boundingBox = self.CascadeClassifierUtils.getBoundingBox(initial_location[0], initial_location[1])
            result = self.camShiftTracker.applyCamShift(image=image, bounding_box=boundingBox, showIO=showIO)
        else:
            # normal call should be this, when we are already initialized
            result = self.camShiftTracker.applyCamShift(image=image, showIO=showIO)
        # it means that we got a valid result!
        if result is not None:
            x = result[0][0]  # get center X
            y = result[0][1]  # get center y
            return (x, y)
        return None

    def getHandViaHaarCascade(self, image, showIO=False):
        return self.CascadeClassifierUtils.getHandViaHaarCascade(image, showIO)

    # Utility to reset the camshift tracker
    def resetCamShift(self):
        self.camShiftTracker.reset()
