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

        self.movingAvgX = MovingAverage(5)
        self.movingAvgY = MovingAverage(5)

    def move(self, x, y, points=0):
        if x is None:
            return

        x = self.movingAvgX.next(x)
        y = self.movingAvgY.next(y)

        mouseX, mouseY = pyautogui.position()
        screenX = (x / self.height) * self.sizeX
        screenY = (y / self.width) * self.sizeY
        dx = screenX - mouseX  # flip image..
        dy = screenY - mouseY

        kp = 0.8

        pyautogui.moveRel(dx * kp, dy * kp)
        if (points != self.pointsT1m):
            if points < 2:
                pyautogui.click(button='left')
                print('Mouse click!!')
            self.pointsT1m = points

    def release(self):
        pass
        # pyautogui.mouseUp(button='right')


from collections import deque


class MovingAverage(object):
    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.queue = deque(maxlen=size)

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.queue.append(val)
        return sum(self.queue) / len(self.queue)
