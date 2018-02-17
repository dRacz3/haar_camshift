import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorKNN(history= 500, dist2Threshold = 100)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',dst)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    edges = cv2.Canny(fgmask,50,100)
    cv2.imshow('edge',edges)


    plt.show()


cap.release()
cv2.destroyAllWindows()
