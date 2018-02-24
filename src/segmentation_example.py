import numpy as np
import cv2

import logging

def nothing(x):
    pass

def applyConvexHull(inputimg, originalImg = None, showIO = False):
    #COUNTOUR DETECTION
    frame = originalImg
    img, contours, hierarchy = cv2.findContours(inputimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape,np.uint8)

    max_area=0
    for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
    try:
        cnt=contours[ci]
    except Exception as e:
        print(e)
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00

    centr=(cx,cy)
    cv2.circle(img,centr,5,[0,0,255],2)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    cv2.drawContours(drawing,[hull],0,(0,0,255),2)

    cv2.circle(frame,centr,5,[0,0,255],2)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    cv2.drawContours(drawing,[hull],0,(0,0,255),2)

    cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt,returnPoints = False)

    if(1):
               defects = cv2.convexityDefects(cnt,hull)
               mind=0
               maxd=0
               for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    dist = cv2.pointPolygonTest(cnt,centr,True)
                    cv2.line(img,start,end,[0,255,0],2)
                    cv2.line(frame,start,end,[0,255,0],2)

                    cv2.circle(img,far,5,[0,0,255],-1)
                    cv2.circle(frame,far,5,[0,0,255],-1)
               print(i)
               i=0

    if showIO:
        cv2.imshow('convexHUll', frame)

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

bgSubThreshold = 100
bgModel = cv2.createBackgroundSubtractorKNN(150, bgSubThreshold )

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_no_bg = removeBackground(bgModel,frame, showIO = True)
    # Converting to HUE-SATURATION image
    #segmentedImg = applySegmentationBasedonHSV(frame, showIO = True)

    cv2.imshow("skinMask", skinMask)
    segmentedImg = frame
    normal = cv2.cvtColor(segmentedImg, cv2.COLOR_HSV2RGB)
    hsv_rgb_stack = np.hstack((segmentedImg,normal))
    cv2.imshow("hsv-> rgb", hsv_rgb_stack)


    # Making it greyscale
    gray = cv2.cvtColor(normal, cv2.COLOR_RGB2GRAY)
    cv2.imshow('RGB->Gray', gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1, 1)



    #thresholded_img = gray_treshold(gray)
    cv2.imshow(th_window_name, gray)

    kernel_close = np.ones((1,1),np.uint8)
    kernel = np.ones((1,1),np.uint8)
    kernel_erode = np.ones((5,5),np.uint8)
    erode = cv2.erode(gray, kernel_erode, iterations  = 3)
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    #valami = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    try:
        applyConvexHull(gray, originalImg = frame ,showIO = True)
    except Exception as e:
        print(e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
