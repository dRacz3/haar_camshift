import cv2
import numpy as np
import time
import pyautogui


class background_remover():
    def __init__(self, binaryTreshold = 60, subTreshold = 700):
        #Background subtraction
        self.threshold = binaryTreshold  #  BINARY threshold
        self.bgSubThreshold = subTreshold
        self.bgModel = cv2.createBackgroundSubtractorKNN(0, self.bgSubThreshold)

    #This function removes the background from a given frame via the KNN algorithm
    def remove_bg(self, frame):
        fgmask = self.bgModel.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 15)
        cv2.imshow('fgmask', fgmask)

        frame_without_bg = cv2.bitwise_and(frame, frame, mask=fgmask)
        return frame_without_bg

class camshift_tracker():
    def __init__(self, initial_frame):
        # setup initial location of window
        r,h,c,w = 250,90,125,125  # simply hardcoded the values
        self.track_window = (c,r,w,h)
        # set up the ROI for tracking # -> bounding box
        roi = initial_frame[r:r+h, c:c+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 10 )

    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        # apply meanshift to get the new location
        self.ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)
        drawing = self.draw_on_img(frame)
        return drawing

    def draw_on_img(self,frame):
        # Draw it on image
        pts = cv2.boxPoints(self.ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        return img2

class haar_classifier():
    def __init__(self, cascPath):
        self.Cascade = cv2.CascadeClassifier(cascPath)

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.Cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=26,
            minSize=(10, 40),
            maxSize = (120, 300),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
        )
        cv2.imshow('gray_haar', self.draw_on_img(gray, results))
        return results

    def draw_on_img(self,frame, results):
        for (x, y, w, h) in results:
            cv2.circle(frame,(int(x + w/2) ,int(y + h/2)), 10, (0,0,255), -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame


def applyConvexHull(inputimg, originalImg = None, showIO = False):
    #COUNTOUR DETECTION
    frame = originalImg
    inputimg = cv2.bitwise_not(inputimg)
    img, contours, hierarchy = cv2.findContours(inputimg,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape,np.uint8)

    max_area=0
    for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
    cnt=contours[ci]
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

    if showIO:
        cv2.imshow('convexHUll', frame)

        hsv_tuning = 'Tuner'
        th_window_name = 'Threshold_tuner'

    return cx,cy, i



class mouseMotionManager():
    def __init__(self, frame):
        self.sizeX, self.sizeY = pyautogui.size()
        self.height, self.width, channels = frame.shape
        self.pointsT1m = 0

        self.x_offset = 0.4 * self.sizeX
        self.y_offset = 0.4 * self.sizeY

    def calc_mapped_values(self, x,y):
        x_new = (x + self.x_offset) / (1 + self.x_offset)
        y_new = (y + self.y_offset) / (1 + self.y_offset)
        x_new *= self.sizeX
        y_new *= self.sizeY
        return x_new, y_new

    def move(self, x, y, points = 0):
        mouseX, mouseY = pyautogui.position()
        screenX = (x/self.height) * self.sizeX
        screenY = (y/self.width) * self.sizeY
#        screenX, screenY = self.calc_mapped_values(x,y)

#        print("before: ({0}|{1}) -> after :({2}|{3})".format(x,y,screenX,screenY))
        dx = (self.sizeX-screenX)-mouseX #flip image..
        dy = screenY-mouseY

        kp = 0.8

        pyautogui.moveRel(dx*kp, dy*kp)
        if (points != self.pointsT1m):
            if points < 2:
                pyautogui.click(button='left')
                print('Mouse click!!')
            self.pointsT1m = points

    def release(self):
        pass
        #pyautogui.mouseUp(button='right')
