import numpy as np
from scipy import stats

class extract:
    def __init__(self,data):
        self.min = np.min(data)
        self.max = np.max(data)
        self.SD = np.std(data)
        self.AM = np.mean(data)
        self.RMS = np.sqrt(np.mean(data**2))
        self.skew = stats.skew(data)
        self.kurt = stats.kurtosis(data)
        self.med = np.median(data)
    
    def get_features(self):
        return np.array([self.min,
                         self.max,
                         self.SD,
                         self.AM,
                         self.RMS,
                         self.skew,
                         self.kurt,
                         self.med])