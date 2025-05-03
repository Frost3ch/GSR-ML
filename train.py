import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import scipy as sp
import extract
import matplotlib.pyplot as plt

class train:
    def __init__(self):
        pass

    def loadArray(self,path):
        data = np.genfromtxt(f'DATA/{path}', delimiter=' ')
        return data
    
    def features(self,data):
        ext = extract.extract(data)
        return ext.get_features()

    def plot(self,data):
        x = np.linspace(0,20,len(data))
        plt.plot(x,data)
        plt.show()

t = train()
data = t.loadArray('1746268944.838717.txt')
features = t.features(data)
t.plot(data)
print(features)