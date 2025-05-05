import numpy as np
import scipy as sp
import extract
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# from sklearn import svm
import joblib

class train:
    def __init__(self):
        pass

    def plot(self,dSet):
        for data in dSet:
            x = np.linspace(0,20,len(data))
            plt.plot(x,data)
        plt.show()

    def train(self,fSet,vSet):
        print('Creating Regressor') 
        clf = RandomForestRegressor()
        # clf = svm.SVC()
        print('Fitting Data...')
        clf.fit(fSet,vSet)
        print('COMPLETE!')
        return clf
    
t = train()
ext = extract.extract()
dSet,fSet = ext.loadFeatures(folder='DATA/GSR')
vSet = ext.loadValAr('DATA/valAr.csv')
# t.plot(dSet)
clf = t.train(fSet,vSet)

joblib.dump(clf,"RFR.pkl")