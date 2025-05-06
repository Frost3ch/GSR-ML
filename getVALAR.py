import numpy as np
import glob

class extract:
    def __init__(self):
        self.sepDist=200
        self.minSamples=50
        
    def load(self):
        paths = glob.glob('DATA/annotations/*.txt')

        all_valArs = []

        for path in paths:
            mat = np.loadtxt(path)
            section = []
            b = mat[1:][1:]-mat[:-1][1:]
            v = abs(b)>self.sepDist
            borders = v.nonzero()[0]+1
            counter=0                                                                                                                                                     
            for i,b in enumerate(borders):
                if i==0:
                    bTimes = [mat[0],mat[b][0]]
                    valAr = np.mean(mat[:b],axis=0)[1:]
                else:
                    bTimes = [borders[i-1],mat[b][0]]
                    valAr = np.mean(mat[borders[i-1]:b],axis=0)[1:]
                if b-borders[i-1]>self.minSamples:
                    section.append([bTimes,valAr])
                    counter+=1
            print(len(borders))
            print(counter)
            all_valArs.append(section)
        #return all_valArs

e = extract()
e.load()