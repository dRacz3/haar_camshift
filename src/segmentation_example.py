import numpy as np
import cv2

import logging

from image_proc_utils import applyConvexHull, mouseMotionManager

def nothing(x):
    pass


hsv_tuning = 'Tuner'
th_window_name = 'Threshold_tuner'

def create_tuner():
# create trackbars for color change
    cv2.namedWindow(hsv_tuning)
    cv2.createTrackbar('H_MIN',hsv_tuning,0,255,nothing)
    cv2.createTrackbar('S_MIN',hsv_tuning,171,255,nothing)
    cv2.createTrackbar('V_MIN',hsv_tuning,144,255,nothing)
    cv2.createTrackbar('H_MAX',hsv_tuning,194,255,nothing)
    cv2.createTrackbar('S_MAX',hsv_tuning,255,255,nothing)
    cv2.createTrackbar('V_MAX',hsv_tuning,251,255,nothing)

    cv2.namedWindow(th_window_name)
    cv2.createTrackbar('Th_min', th_window_name, 0, 255, nothing)
    cv2.createTrackbar('Th_max', th_window_name, 255, 255, nothing)

def color_treshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    hmin = cv2.getTrackbarPos('H_MIN',hsv_tuning)
    smin = cv2.getTrackbarPos('S_MIN',hsv_tuning)
    vmin = cv2.getTrackbarPos('V_MIN',hsv_tuning)
    hmax = cv2.getTrackbarPos('H_MAX',hsv_tuning)
    smax = cv2.getTrackbarPos('S_MAX',hsv_tuning)
    vmax = cv2.getTrackbarPos('V_MAX',hsv_tuning)

    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    return res

def gray_treshold(img):
    th_min = cv2.getTrackbarPos('Th_min', th_window_name)
    th_max = cv2.getTrackbarPos('Th_max', th_window_name)
    ret, thresh = cv2.threshold(img, th_min, th_max, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def removeBackground(backgroundModel, image, showIO = False):
    #Removing background as much as possible
    fgmask = backgroundModel.apply(image)
    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 1, 1)
    #cv2.imshow('No BG', fgmask)
    res = cv2.bitwise_and(image, image, mask=fgmask)
    if showIO:
        stack = np.hstack((image,res))
        cv2.imshow('removeBackground', stack)
    return res

# Apply thresholding to leave MOSTLY skin
def applySegmentationBasedonHSV(img, showIO = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blurring it a bit
    blurred = cv2.GaussianBlur(hsv, (7, 7), 1, 1)
    blurred = color_treshold(hsv)
    if showIO:
        stack = np.hstack((img,blurred))
        cv2.imshow('applySegmentationBasedonHSV', stack)
    return blurred

###########################x MAIN CODE x#####################################
create_tuner()
cap = cv2.VideoCapture(0)

bgSubThreshold = 50
bgModel = cv2.createBackgroundSubtractorKNN(1000, bgSubThreshold )

ret, frame = cap.read()
mouseManager = mouseMotionManager(frame)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_no_bg = removeBackground(bgModel,frame, showIO = False)

#    edges = cv2.Canny(frame_no_bg,100,100)
#    cv2.imshow('edges', edges)


    # Converting to HUE-SATURATION image
    segmentedImg = applySegmentationBasedonHSV(frame_no_bg, showIO = False)
    normal = cv2.cvtColor(segmentedImg, cv2.COLOR_HSV2RGB)
    hsv_rgb_stack = np.hstack((segmentedImg,normal))
    #cv2.imshow("hsv-> rgb", hsv_rgb_stack)

    # Making it greyscale
    gray = cv2.cvtColor(frame_no_bg, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 1, 1)
    binary_thresholded = gray_treshold(gray)

    gray_binary_stack = np.hstack((gray,binary_thresholded))
    #cv2.imshow("gray_binary_stack", gray_binary_stack)

    kernel_close = np.ones((1,1),np.uint8)
    kernel = np.ones((1,1),np.uint8)
    kernel_erode = np.ones((5,5),np.uint8)
    erode = cv2.erode(binary_thresholded, kernel_erode, iterations  = 3)
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    #valami = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    try:
        cx, cy, points = applyConvexHull(closing, originalImg = frame ,showIO = False)
        mouseManager.move(cx,cy, points)
    except Exception as e:
        print(e)
        #mouseManager.release()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
mouseManager.release()
cv2.destroyAllWindows()
