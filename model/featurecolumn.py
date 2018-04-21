import random
import numpy as np
import math
from model.featuremcell import FeatureMCell
from lib.helper import Helper

class FeatureColumn:

    FMC_NUM = 100

    def __init__(self, name, sensor):
        self.name = name
        self.sensor = sensor
        self.fmcs = []
        self.initFMC()
    
    def initFMC(self):
        for i in range(0,FeatureColumn.FMC_NUM):
            fmc = FeatureMCell('FMC'+str(i), self.sensor, self)
            self.fmcs.append(fmc)

        #Init links   
        for i in range(0,FeatureColumn.FMC_NUM):
            linksPos = Helper.pos_random_sample(FeatureColumn.FMC_NUM,0.01,0.03)
            fmcs = []
            for pos in linksPos:
                if i != pos:
                    fmcs.append(self.fmcs[pos])
            self.fmcs[i].initFMCConnect(fmcs)            
    
    def run(self):
        for fmc in self.fmcs:
            fmc.run()
        for fmc in self.fmcs:
            fmc.learnSequence()
        for fmc in self.fmcs:
            fmc.predict()
            fmc.debug(lev=1)

    def getPredictFmc(self):
        predMap = [fmc.isNextActiveScore for fmc in self.fmcs]
        sortindex = np.argsort(predMap)
        sortindex = sortindex[:90]
        for idx in sortindex:
            predMap[idx] = 0.0
        #print(predMap)
        x = int(np.sqrt(FeatureColumn.FMC_NUM))
        return np.reshape(predMap,(x,x))
    
    def getFeatureMap(self):       
        fmcs = self.fmcs
        w = np.sqrt(FeatureColumn.FMC_NUM)
        rowa = []
        ret = []
        for i in range(0,FeatureColumn.FMC_NUM):
            if i % w == 0:
                if i > 0:
                    if i == w:
                        ret = rowa
                    else:
                        ret = np.concatenate((ret, rowa), axis=0) 
                rowa = fmcs[i].getFeatureImg(border=True)
            else:
                rowa = np.concatenate((rowa, fmcs[i].getFeatureImg(border=True)), axis=1)
            
        return ret