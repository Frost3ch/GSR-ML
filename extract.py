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
        self.sepDist=200
        self.minSamples=50
        
    def loadValar(self):
        paths = glob.glob('DATA/annotations/*.txt')

        all_valArs = []

        for path in paths:
            mat = np.loadtxt(path)
            section = []
            b = mat[1:][1:]-mat[:-1][1:]
            v = abs(b)>self.sepDist
            borders = v.nonzero()[0]+1
            counter=0                                                                                                                                                     
            for i,b in enumerate(borders):
                if i==0:
                    bTimes = [mat[0][0],mat[b][0]]
                    valAr = np.mean(mat[:b],axis=0)[1:]
                else:
                    bTimes = [borders[i-1][0],mat[b][0]]
                    valAr = np.mean(mat[borders[i-1]:b],axis=0)[1:]
                if b-borders[i-1]>self.minSamples:
                    section.append([bTimes,valAr])
                    counter+=1
            # print(len(borders))
            # print(counter)
            all_valArs.append(section)
        return all_valArs
    
    def get_ECG_features(self,data):
        peaks, _ = find_peaks(data, distance=80)
        rr_intervals = np.diff(peaks) / 256 # in seconds
        heart_rate = 60 / np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        diff_rr = np.abs(np.diff(rr_intervals))
        pnn50 = np.sum(diff_rr > 0.05) / len(diff_rr) * 100  # 0.05s = 50ms
                                  
        return np.array([
                         heart_rate,
                         sdnn,
                         rmssd,
                         pnn50
                         ])
    
    def get_GSR_features(self,data):
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
    
    def loadData(self,folder):
        paths = glob.glob(f'{folder}/physiological/*.txt')

        t_fSet = []
        for path in paths:
            mat = np.loadtxt(path)
            mat = np.hsplit(mat,9)
            fSet = [mat[0].flatten(),mat[4].flatten(),mat[1].flatten()]    
            t_fSet.append(fSet)
        return t_fSet
    
    def pairXY(self,t_fSet,all_valArs):
        fX = []
        fY = []
        for i,valAr in enumerate(all_valArs):
            j = 0
            sect = []
            while valAr[0][1] > t_fSet[i][0][j]:
                j+=1
            GSR_fSet = self.get_GSR_features(t_fSet[i][1][:j])
            ECG_fSet = self.get_GSR_features(t_fSet[i][2][:j])

            fX.append(np.concatenate((GSR_fSet,ECG_fSet),axis=1))
            fY.append(valAr[1])
        return fX,fY