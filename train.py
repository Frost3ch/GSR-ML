import numpy as np
import scipy as sp
import extract
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

class train:
    def __init__(self):
        pass

    def plot(self,dSet):
        for data in dSet:
            x = np.linspace(0,20,len(data))
            plt.plot(x,data)
        plt.show()

    def train(self,fSet,vSet):
        clf = RandomForestClassifier()
        clf.fit(fSet,vSet)
        return clf
    

t = train()
dSet,fSet,vSet = extract.extract().loadFeatures(folder='DATA')
t.plot(dSet)
clf = t.train(fSet,vSet)
