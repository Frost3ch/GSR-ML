import serial
import numpy as np
import time
import csv
import read
import extract
import joblib


print('collecting data...')

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
fSet = extract.extract().get_features(np.array(data))
clf = joblib.load("model.pkl")
predV = clf.predict([fSet])
print(predV)