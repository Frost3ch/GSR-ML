import serial
import numpy as np
import time
import csv

class arduino:
    def __init__(self):
        self.arduino = serial.Serial(port='COM3',baudrate=9600,timeout=0.2)

    def read(self):
        try:
            return int(self.arduino.readline().strip())
        except:
            return None

arduino = arduino()

while arduino.read()==None:
    pass

while True:
    cT = time.time() 
    data = []
    while time.time()-cT <= 20:
        p = arduino.read()
        print(p)
        data.append(p)
    with open(f"DATA/{cT}.txt","a") as f:
        csv.writer(f,delimiter=' ').writerow(data)