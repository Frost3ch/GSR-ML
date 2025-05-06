import numpy as np
import scipy as sp
import extract
import ecg as ecgClass
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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

#Extract GSR data 
t = train()
ext = extract.extract()
_,gsr_fSet = ext.loadFeatures(folder='DATA/GSR_TRAIN')

#Extract ECG data
ecg = ecgClass.extract()
ecg_dSet, ecg_fSet = ecg.loadFeatures(folder='DATA/ECG_TRAIN')
# x = np.linspace(0,20,len(ecg_dSet[0]))
# plt.plot(x,ecg_dSet[0])
# plt.show()

# print(gsr_fSet)
# print(ecg_fSet)
fSet = np.concatenate((gsr_fSet,ecg_fSet),axis=1)
# print(fSet)
vSet, aSet = ext.loadValAr('DATA/valAr_TRAIN.csv')
v_clf = t.train(fSet,vSet)
a_clf = t.train(fSet,aSet)

#Test model with test data
_, gsr_fSet_test = ext.loadFeatures(folder='DATA/GSR_TEST')
_, ecg_fSet_test = ecg.loadFeatures(folder='DATA/ECG_TEST')
vSet_test, aSet_test = ext.loadValAr('DATA/valAr_TEST.csv')
fSet_test = np.concatenate((gsr_fSet_test,ecg_fSet_test),axis=1)
vSet_pred = v_clf.predict(fSet_test)
aSet_pred = a_clf.predict(fSet_test)
# print(vSet_pred)
v_accuracy =  r2_score(vSet_test,vSet_pred)
a_accuracy =  r2_score(aSet_test,aSet_pred)
print(f"Valence Accuracy: {v_accuracy}")
print(f"Arousal Accuracy: {a_accuracy}")


joblib.dump(v_clf,"RFR_V.pkl")
joblib.dump(a_clf,"RFR_A.pkl")