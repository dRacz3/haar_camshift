import cv2
import numpy as np


class GestureDetector():
    def __init__(self):
        pass

    def processArea(self, img):
        cv2.imshow('kep', img)

    def contourFinder(self, img, camshift_result=None):
        try:
            gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cis = self.get_camshifted_section(img, camshift_result)
            gimg = cv2.cvtColor(cis, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(gimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12)
            #result = cv2.bitwise_and(img, img, mask=thresh)
            im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

            maxArea = 0
            secondMax = 0
            maxContour = contours[0]
            secondContour = contours[0]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > maxArea:
                    secondMax = maxArea
                    secondContour = maxContour
                    maxArea = area
                    maxContour = cnt

            M = cv2.moments(maxContour)
            rect = cv2.minAreaRect(maxContour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(cis, [box], 0, (0, 0, 255), 2)

            M = cv2.moments(secondContour)
            rect = cv2.minAreaRect(secondContour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(cis, [box], 0, (0, 0, 255), 2)

            #a = np.hstack((img, result))
            #cv2.imshow("ASD", a)
            cv2.imshow("cis", cis)
        except Exception as e:
            print(e)

    def get_camshifted_section(self, img, camshift_result):
        x, y = camshift_result[0]
        w, h = camshift_result[1]

        #print('imshape', img.shape())

        stx = int(x - 150)
        if stx < 0:
            stx = 0
        enx = int(stx + 300)
        sty = int(y - 150)
        if sty < 0:
            sty = 0
        eny = int(sty + 300)

        res = img[stx:enx, sty:eny]
        return res


if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    gd = GestureDetector()
    while True:
        ret, frame = camera.read()
        gd.contourFinder(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
