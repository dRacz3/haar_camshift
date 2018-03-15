import logging

import cv2
import numpy as np
from imageOperations import ImageOperations
from mouseMotionManager import mouseMotionManager


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
        result, mask = self.operations.removeBackground(startingImage, showIO=False)
        # result, mask = self.operations.imageThresholding(result, showIO=True) # ->not used
        result, mask = self.operations.adaptiveImageThresholding(result, showIO=False)
        initial_location = None
        if not self.isFound:
            self.logger.debug("Still looking for fingers...")
            fingers_results = self.operations.getHandViaHaarCascade(result, showIO=True)
            # if got results -> check if we got enough markers to say it's a hand
            if fingers_results is not None:
                self.isFound, initial_location = self.operations.CascadeClassifierUtils.evaluateIfHandisFound(
                    fingers_results)
                self.logger.debug(
                    "Finger results contains something, is it enough for to say its a hand? {0}".format(self.isFound))
        # if we got a new location in this round that means that it's the only time when it's not None
        # so we pass it to the camshift, and expect it to initialize itself on this ROI
        # camshift_result = None
        if initial_location is None:
            # Mostly  this should be called
            camshift_result = self.operations.applyCamShift(result, showIO=True)
        else:
            # Initializer calls only
            self.logger.info("Found new initial locations..should reinitialize camshift!")
            camshift_result = self.operations.applyCamShift(result, initial_location)
        if camshift_result is not None:
            self.logger.debug('HAND LOCATION: {0}|{1}'.format(camshift_result[0], camshift_result[1]))
            self.mouseMotionManager.move(camshift_result[0], camshift_result[1])
            asd = np.array(startingImage)
            # Move mouse to location: camshift_result[0], camshift_result[1]!
        else:
            # if we got back nothing, it means we lost track of the object, we need to find it again via cascade
            self.isFound = False
            self.logger.debug("No hand can be detected..")

            # self.operations.color_treshold(result, showIO=True)
            # self.operations.getConvexHulls(result, mask, showIO=True)
        return result

    def run(self):
        while True:
            self.ProcessOneMoreFrame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


app = program()
app.run()
