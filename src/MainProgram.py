import logging

import cv2
import numpy as np
from imageOperations import ImageOperations
from mouseMotionManager import mouseMotionManager
from gestureDetector import GestureDetector


class program(object):
    def __init__(self):
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('program')
        self.logger.setLevel('INFO')
        self.logger.info("Program has been started!")
        self.camera = cv2.VideoCapture(0)
        self.operations = ImageOperations()
        self.isFound = False

        ret, frame = self.camera.read()
        self.mouseMotionManager = mouseMotionManager(frame)

        self.gestureDetector = GestureDetector()

    def release(self):
        # When everything done, release the capture
        self.logger.info("Program has finished!")
        self.camera.release()
        cv2.destroyAllWindows()

    # process one frame
    def ProcessOneMoreFrame(self):
        ret, frame = self.camera.read()
        frame = self.operations.flipImage(frame)
        result = self.process(frame)

        compareResult = np.hstack((frame, result))
        cv2.imshow('Result frame', compareResult)

    def process(self, startingImage):
        result = cv2.bitwise_and(startingImage, startingImage)
        enableFrames = True
        backgroundRemovalResult, mask = self.operations.removeBackground(startingImage, showIO=enableFrames)
        adaptiveImageThresholdingResult, mask = self.operations.adaptiveImageThresholding(
            backgroundRemovalResult, showIO=False)
        initial_location = None
        if not self.isFound:
            self.logger.debug("Still looking for fingers...")
            handResults = self.operations.getHandViaHaarCascade(
                startingImage, showIO=enableFrames)
            # if got results -> check if we got enough markers to say it's a hand
            if handResults is not None:
                self.isFound, initial_location = self.operations.CascadeClassifierUtils.evaluateIfHandisFound(
                    handResults)
                self.logger.debug(
                    "Finger results contains something, is it enough for to say its a hand? {0}".format(self.isFound))
        # if we got a new location in this round that means that it's the only time when it's not None
        # so we pass it to the camshift, and expect it to initialize itself on this ROI
        # camshift_result = None
        if initial_location is None:
            # Mostly  this should be called
            camshift_result = self.operations.applyCamShift(adaptiveImageThresholdingResult, showIO=enableFrames)
        else:
            # Initializer calls only
            self.logger.info("Found new initial locations..should reinitialize camshift!")
            camshift_result = self.operations.applyCamShift(adaptiveImageThresholdingResult, initial_location)
        if camshift_result is not None:
            self.logger.debug('HAND LOCATION: {0}|{1}'.format(camshift_result[0][0], camshift_result[0][1]))
            ret = self.operations.CascadeClassifierUtils.getFaceViaHaarCascade(startingImage, showIO=enableFrames)
            if ret is not None:
                try:
                    manhattan_distance = self.operations.calculate_manhattan_distance(ret, camshift_result)
                    face_box_width = ret[0][3]
                    if manhattan_distance < face_box_width:
                        self.isFound = False
                        self.operations.resetCamShift()
                        self.logger.info("Camshift result probably stuck on face, dropping current detection!")
                except Exception as e:
                    self.logger.debug(e)
            self.mouseMotionManager.move(camshift_result[0][0], camshift_result[0][1])
        else:
            # if we got back nothing, it means we lost track of the object, we need to find it again via cascade
            self.isFound = False
            self.logger.debug("No hand can be detected..")
        return result

    def run(self):
        while True:
            self.ProcessOneMoreFrame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


app = program()
app.run()
