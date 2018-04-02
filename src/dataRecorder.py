import pandas as pd
import numpy as np

import time


class dataRecorder(object):
    def __init__(self):
        self.camshift = {'time': [], 'x': [], 'y': []}
        self.cascade = {'time': [], 'x': [], 'y': []}

    def recordCamshiftPosition(self, x, y):
        ts = time.time()
        self.camshift['time'].append(ts)
        self.camshift['x'].append(x)
        self.camshift['y'].append(y)

    def recordHaarPosition(self, x, y):
        ts = time.time()
        self.cascade['time'].append(ts)
        self.cascade['x'].append(x)
        self.cascade['y'].append(y)

    def save(self):
        df = pd.DataFrame(self.cascade)
        df.to_csv('cascade.csv')
        df = pd.DataFrame(self.camshift)
        df.to_csv('camshift.csv')
