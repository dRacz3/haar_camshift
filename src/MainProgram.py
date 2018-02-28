import cv2
import numpy as np
import logging
from imageOperations import ImageOperations


class program(object):
    def __init__(self):
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('program')
        self.logger.setLevel('DEBUG')
        self.logger.info("Program has been started!")
        self.camera = cv2.VideoCapture(0)
        self.operations = ImageOperations()
        self.isFound = False

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
        result, mask = self.operations.removeBackground(startingImage, showIO=True)
        # result, mask = self.operations.imageThresholding(result, showIO=True) # ->not used
        result, mask = self.operations.adaptiveImageThresholding(result, showIO=True)
        fingers_results = self.operations.getHandViaHaarCascade(result, showIO=True)
        if not self.isFound and fingers_results is not None:
            self.logger.info("fingers not found yet, finger results aint none either")
            self.isFound, initial_location = self.operations.evaluateIfHandisFound(fingers_results)
        # if self.isFound:
            #self.logger.info("fingers found! goin to cam shift! isFound? :{0}".format(self.isFound), )
            #self.operations.applyCamShift(result, initial_location)
        #self.operations.color_treshold(result, showIO=True)
        #self.operations.getConvexHulls(result, mask, showIO=True)
        return result

    def run(self):
        while True:
            self.ProcessOneMoreFrame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


app = program()
app.run()
