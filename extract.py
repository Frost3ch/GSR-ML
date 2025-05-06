import numpy as np
from scipy import stats
import glob
import csv
from scipy import io
import os
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class extract:
    def __init__(self):
        pass
    
    def get_features(self,data):
        min = np.min(data)
        max = np.max(data)
        SD = np.std(data)
        AM = np.mean(data)
        var = np.var(data)
        RMS = np.sqrt(np.mean(data**2))
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        med = np.median(data)

        peaks, _ = find_peaks(data, distance = 150, prominence=1.0)  # Adjust height threshold as needed
        num_peaks = len(peaks)
        mean_peak_height = np.mean(data[peaks]) if len(peaks) > 0 else 0
        max_peak_height = np.max(data[peaks]) if len(peaks) > 0 else 0

        # print(peaks)
        # print(num_peaks)
        # print(mean_peak_height)

        return peaks,np.array([
                        #  min,
                        #  max,
                         SD,
                         AM,
                        #  var,
                        #  RMS,
                        #  skew,
                        #  kurt,
                        #  med,
                         num_peaks,
                         mean_peak_height
                        #  max_peak_height
                         ])
    
    
    def loadFeatures(self,folder):
        data_paths = sorted(glob.glob(f'{folder}/*.mat'))
        d_set = []
        f_set = []
        for path in data_paths:
            data = io.loadmat(path)
            data = data['GSRdata']
            data =  np.array(data).flatten()
            data = savgol_filter(data, window_length=11, polyorder=2)
            peaks,features = self.get_features(data)
            x = np.linspace(0,len(data),len(data))
            # print(path)
            # plt.plot(x,data)
            # plt.plot(peaks, data[peaks], "x")
            # plt.show()
            d_set.append(data)
            f_set.append(features)
        return d_set,f_set
    
    def loadValAr(self,path):
        v_set = []
        a_set = []
        with open(path) as f:
            reader = csv.reader(f,delimiter='\t')
            data = list(reader)
            data = [[int(x),int(y)] for [x,y] in data]
        data = np.hsplit(np.array(data),2)
        v_set = np.ravel(data[0])

        a_set = np.ravel(data[1])
        return v_set,a_set