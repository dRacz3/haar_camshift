import cv2
import numpy as np
import time
import pyautogui

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
        if x is None:
            return
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
