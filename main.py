import serial
import numpy as np
import time
import csv
import read
import extract
import joblib
import matplotlib.pyplot as plt

print('collecting data...')

#Collect Data
arduino = read.arduino()
while arduino.read()==None:
    pass

cT = time.time() 
data = []
while time.time()-cT <= 20:
    p = arduino.read()
    if p!=None:
        print(p)
        data.append(p)

#Extract Features and predict Valence and Arousal levels
fSet = extract.extract().get_features(np.array(data))
clf = joblib.load("RFR.pkl")
predV = clf.predict([fSet])
print(predV)

#Plot Valence and Arousal Levels
ax = plt.axes()
ax.spines[['top', 'right']].set_visible(False)
ax.spines.left.set_position('zero')
ax.spines.bottom.set_position('zero')

plt.plot(predV[0][0]-5,predV[0][1]-5,'r+')
plt.axis([-5, 5, -5, 5])
plt.show()