import serial
import numpy as np
import time
import csv

class arduino:
    def __init__(self):
        self.arduino = serial.Serial(port='COM3',baudrate=9600,timeout=0.05)

    def read(self):
        try:
            return int(self.arduino.readline().strip())
        except:
            return None

def main():
    arduino = arduino()

    while arduino.read()==None:
        pass

    while True:
        cT = time.time() 
        data = []
        while time.time()-cT <= 20:
            p = arduino.read()
            if p!=None:
                print(p)
                data.append(p)
        valAr = [float(x) for x in input("enter valence + arousal (e.g. 9 -2): ").split()]
        data.append(valAr[0])
        data.append(valAr[1])
        with open(f"DATA/{cT}.csv","a") as f:
            csv.writer(f,delimiter=' ').writerow(data)

if __name__ == "__main__":
    main()

