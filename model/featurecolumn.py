import random
from model.featuremcell import FeatureMCell
import numpy as np
import math

class FeatureColumn:

    FMC_NUM = 100

    def __init__(self, name, sensor):
        self.name = name
        self.sensor = sensor
        self.fmcs = []
        self.initFMC()
    
    def initFMC(self):
        for i in range(0,FeatureColumn.FMC_NUM):
            fmc = FeatureMCell('FMC'+str(i), self.sensor)
            self.fmcs.append(fmc)

    def run(self):
        for fmc in self.fmcs:
            fmc.run()
            fmc.debug()
    
    def getSortedFMC(self):
        return sorted(self.fmcs, key=lambda x: x.activeFrq, reverse=True)
    
    def getFeatureMap(self,srt=False):
        if srt == True:
            fmcs = self.getSortedFMC()
        else:
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

    def output(self):
        print("out")

    def initFmcs(self):
        f = self.fmcs