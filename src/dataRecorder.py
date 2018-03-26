import pandas as pd
import numpy as np

class dataRecorder(object):
    def __init__(self):
        self.data = {'cx':[], 'cy':[], 'cascadeX': [], 'cascadeY' : []}

    def recordCamshiftPosition(self, x, y):
        self.data['cx'].append(x)
        self.data['cy'].append(y)

    def recordHaarPosition(self,x,y):
        self.data['cascadeX'].append(x)
        self.data['cascadeY'].append(y)

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv('asd.csv')
        self.data = {'cx':[], 'cy':[], 'cascadeX': [], 'cascadeY' : []}
