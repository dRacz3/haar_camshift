import cv2
import numpy as np
import logging

class program(object):
    def __init__(self):
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('program')
        self.logger.setLevel('DEBUG')
        self.logger.info("Program has been started!")
        self.camera = cv2.VideoCapture(0)


    def release(self):
        # When everything done, release the capture
        self.logger.info("Program has finished!")
        self.camera.release()
        cv2.destroyAllWindows()

    #process one frame
    def ProcessOneMoreFrame(self):
        ret, frame = self.camera.read()
        result = self.process(frame)

        compareResult = np.hstack((frame,result))
        cv2.imshow('Result frame', compareResult)



    def process(self, startingImage):
        result = startingImage
        return result


    def run(self):
        while True:
            self.ProcessOneMoreFrame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

app = program()
app.run()
