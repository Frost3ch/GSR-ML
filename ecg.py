import numpy as np
from scipy import stats
import glob
import csv
from scipy import io
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class extract:
    def __init__(self):
        pass
    
    def get_features(self,data):
        peaks, _ = find_peaks(data, distance=80)
        rr_intervals = np.diff(peaks) / 256 # in seconds
        heart_rate = 60 / np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        diff_rr = np.abs(np.diff(rr_intervals))
        pnn50 = np.sum(diff_rr > 0.05) / len(diff_rr) * 100  # 0.05s = 50ms

        # print(heart_rate)

                                  
        return np.array([
                         heart_rate,
                         sdnn,
                         rmssd,
                         pnn50
                         ])
    
    
    def loadFeatures(self,folder):
        data_paths = sorted(glob.glob(f'{folder}/*.mat'))
        d_set = []
        f_set = []
        for path in data_paths:
            data = io.loadmat(path)
            data = data['ECGdata']
            data =  np.array(data).flatten()
            features = self.get_features(data)
            d_set.append(data)
            f_set.append(features)
        return d_set, f_set