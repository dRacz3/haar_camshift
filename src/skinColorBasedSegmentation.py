import logging

import cv2
import numpy as np
from imageOperations import ImageOperations
from objectDetection import colorBasedSegmenter
from dataRecorder import dataRecorder


class program(object):
    def __init__(self):
        FORMAT = '[%(asctime)-15s][%(levelname)s][%(funcName)s] %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('program')
        self.logger.setLevel('INFO')
        self.logger.info("Program has been started!")
        self.camera = cv2.VideoCapture(0)
        self.operations = ImageOperations()
        self.segmenter = colorBasedSegmenter()

        ret, frame = self.camera.read()
        self.dataRecorder = dataRecorder()

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
        bgrm , mask = self.operations.removeBackground(startingImage, erosion_kernel_size = 15)
        result = self.segmenter.applyColorBasedSegmenetation(bgrm , showIO=True)

        return result

    def run(self):
        while True:
            self.ProcessOneMoreFrame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.dataRecorder.save()
                break


app = program()
app.run()
