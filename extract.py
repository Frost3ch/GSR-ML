import numpy as np
from scipy import stats
import glob
import csv
from scipy import io
import os

class extract:
    def __init__(self):
        pass
    
    def get_features(self,data):
        self.min = np.min(data)
        self.max = np.max(data)
        self.SD = np.std(data)
        self.AM = np.mean(data)
        self.RMS = np.sqrt(np.mean(data**2))
        self.skew = stats.skew(data)
        self.kurt = stats.kurtosis(data)
        self.med = np.median(data)

        return np.array([self.min,
                         self.max,
                         self.SD,
                         self.AM,
                         self.RMS,
                         self.skew,
                         self.kurt,
                         self.med])
    
    # def loadFeatures(self,folder):
    #     data_paths = glob.glob(f'{folder}/*.csv')
    #     d_set = []
    #     f_set = []
    #     v_set = []
    #     for path in data_paths:
    #         with open(path) as f:
    #             reader = csv.reader(f)
    #             data = list(reader)[0][0].split()
    #             data = [float(x) for x in data]

    #         ar = data.pop()
    #         val = data.pop()
    #         valAr = [val,ar]
    #         print(len(data))
    #         data = np.array(data)
    #         features = self.get_features(data)
    #         d_set.append(data)
    #         f_set.append(features)
    #         v_set.append(valAr)
    #     return d_set,f_set,v_set
    
    def loadFeatures(self,folder):
        data_paths = glob.glob(f'{folder}/*.mat')
        d_set = []
        f_set = []
        for path in data_paths:
            data = io.loadmat(path)
            data = data['s'][0]
            data = np.array(data)
            features = self.get_features(data)
            d_set.append(data)
            f_set.append(features)
        return d_set,f_set
    
    def loadValAr(self,path):
        v_set = []
        with open(path) as f:
            reader = csv.reader(f)
            data = list(reader)
            data = [[int(x),int(y)] for [x,y] in data[1:]]
        v_set = np.array(data)
        return v_set