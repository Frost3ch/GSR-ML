import numpy as np
import scipy as sp
import extract
import matplotlib.pyplot as plt

class train:
    def __init__(self):
        pass

    def plot(self,dSet):
        for data in dSet:
            x = np.linspace(0,20,len(data))
            plt.plot(x,data)
        plt.show()

t = train()
dSet,fSet,vSet = extract.extract().loadFeatures(folder='DATA')
print(dSet[0])
print(fSet[0])
print(vSet[0])

t.plot(dSet)