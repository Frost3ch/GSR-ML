import numpy as np
import scipy as sp
import extract
import hide.ecg as ecgClass
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
    


t = train()
ext = extract.extract()
t_fSet = ext.loadData('DATA')
all_valArs = ext.loadValar()
fX,fY = ext.pairXY(t_fSet,all_valArs)

# x = np.linspace(0,20,len(ecg_dSet[0]))
# plt.plot(x,ecg_dSet[0])
# plt.show()

fY = np.hsplit(fY,2)
vals = fY[0].flatten()
arous = fY[1].flatten()

v_clf = t.train(fX,vals)
a_clf = t.train(fX,arous)

#Test model with test data
t_fSet = ext.loadData('TEST')
all_valArs = ext.loadValar()
fX,fY = ext.pairXY(t_fSet,all_valArs)
fY = np.hsplit(fY,2)
vals = fY[0].flatten()
arous = fY[1].flatten()
vSet_pred = v_clf.predict(fX)
aSet_pred = a_clf.predict(fX)
# print(vSet_pred)
v_accuracy =  r2_score(vals,vSet_pred)
a_accuracy =  r2_score(arous,aSet_pred)
print(f"Valence Accuracy: {v_accuracy}")
print(f"Arousal Accuracy: {a_accuracy}")


joblib.dump(v_clf,"RFR_V.pkl")
joblib.dump(a_clf,"RFR_A.pkl")