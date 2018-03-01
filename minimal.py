import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def nothing(x):
    pass


cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('h_min', 'image', 0, 255, nothing)
cv2.createTrackbar('s_min', 'image', 0, 255, nothing)
cv2.createTrackbar('v_min', 'image', 0, 255, nothing)
cv2.createTrackbar('h_max', 'image', 255, 255, nothing)
cv2.createTrackbar('s_max', 'image', 255, 255, nothing)
cv2.createTrackbar('v_max', 'image', 255, 255, nothing)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hmi = cv2.getTrackbarPos('h_min', 'image')
    smi = cv2.getTrackbarPos('s_min', 'image')
    vmi = cv2.getTrackbarPos('v_min', 'image')
    hma = cv2.getTrackbarPos('h_max', 'image')
    sma = cv2.getTrackbarPos('h_max', 'image')
    vma = cv2.getTrackbarPos('h_max', 'image')

    # define range of blue color in HSV
    lower_blue = np.array([hmi, smi, vmi])
    upper_blue = np.array([hma, sma, vma])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
