import pandas as pd
import numpy as np

class dataRecorder(object):
    def __init__(self):
        self.camshift = {'x':[], 'y':[]}
        self.cascade = {'x': [], 'y' : []}

    def recordCamshiftPosition(self, x, y):
        self.camshift['x'].append(x)
        self.camshift['y'].append(y)

    def recordHaarPosition(self,x,y):
        self.cascade['x'].append(x)
        self.cascade['y'].append(y)

    def save(self):
        df = pd.DataFrame(self.cascade)
        df.to_csv('cascade.csv')
        df = pd.DataFrame(self.camshift)
        df.to_csv('camshift.csv')
