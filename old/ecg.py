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
        paths = glob.glob('DATA/physiological/*.txt')

        t_readings = []
        for path in paths:
            mat = np.loadtxt(path)
            reading = []
            mat = np.hsplit(mat,9)   
            reading = [mat[0],mat[1],mat[3]]    
            t_readings.append(reading)
        return t_readings