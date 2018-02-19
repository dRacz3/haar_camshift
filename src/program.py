import numpy as np
import cv2 as cv
import time
from image_proc_utils import background_remover, camshift_tracker, haar_classifier
cap = cv.VideoCapture(0)
# take first frame of the video
ret,frame = cap.read()

camshift_tracker = camshift_tracker(frame)
bgrem = background_remover()

handxml = 'haar_hand.xml'
hand_classifier = haar_classifier(handxml)
fingerxml = 'haar_finger.xml'
finger_classifier = haar_classifier(fingerxml)


while(1):
    ret ,frame = cap.read()
    res = bgrem.remove_bg(frame)

    if ret == True:
        result = camshift_tracker.process(res)
        hand_res = hand_classifier.process(res)
        finger_res = finger_classifier.process(res)

        #cv.imshow('Hand result', hand_classifier.draw_on_img(result, hand_res))
        #cv.imshow('Finger result', finger_classifier.draw_on_img(result, finger_res))
        cv.imshow('Camshift Result', result)
        cv.imshow('Background removal', res)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            pass
#            cv.imwrite(chr(k)+".jpg",result)
    else:
        break
cv.destroyAllWindows()
cap.release()
