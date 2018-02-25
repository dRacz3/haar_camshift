import cv2
import sys
import pyautogui
import numpy as np

# What fille?
cascPath = 'new_cascade.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 150

bgModel = cv2.createBackgroundSubtractorKNN(4, bgSubThreshold, detectShadows=True)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    fgmask = bgModel.apply(frame)

    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.GaussianBlur(fgmask, (1, 1), 1, 1)
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)

    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    maskStack = np.hstack((closing, gradient))
    cv2.imshow("maskStack", maskStack)
    openStack = np.hstack((fgmask, opening))
    cv2.imshow("opening", openStack)


    gray = cv2.GaussianBlur(closing, (1, 1), 1, 1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(res, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    contour = cv2.Canny(blur,10,40)

    results = faceCascade.detectMultiScale(
        res,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(0, 0),
        maxSize = (500, 500),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    mouseX, mouseY = pyautogui.position()


    # Draw a rectangle around the faces
    for (x, y, w, h) in results:
        sizeX, sizeY = pyautogui.size()
        height, width, channels = frame.shape
        screenX = (x/height) * sizeX
        screenY = (y/width) *sizeY
        dx = (sizeX-screenX)-mouseX #flip image..
        dy = screenY-mouseY

        kp = 0.8

        pyautogui.moveRel(dx*kp, dy*kp)
#        pyautogui.moveTo(sizeX-x*3,y*3)
        cv2.circle(res,(int(x + w/2) ,int(y + h/2)), 10, (0,0,255), -1)
#        cv2.circle(frame, (x+w/2, y+h/2))
        cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

    cv2.imshow('Video', frame)
    #cv2.imshow('Canny', contour)
    cv2.imshow('res', res)
    cv2.imshow('tresh', thresh)
    #cv2.imshow('blur', blur)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
