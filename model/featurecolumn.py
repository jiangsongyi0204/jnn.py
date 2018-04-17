import random
import numpy as np
import math
from model.featuremcell import FeatureMCell

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

        #Init links   
        for i in range(0,FeatureColumn.FMC_NUM):
            posarr = [m for m in range(0,FeatureColumn.FMC_NUM)]
            picksize = random.randrange(round(FeatureColumn.FMC_NUM*0.1), round(FeatureColumn.FMC_NUM*0.3))
            linksPos = random.sample(posarr,picksize)
            fmcs = []
            for pos in linksPos:
                fmcs.append(self.fmcs[pos])
            self.fmcs[i].initFMCLinks(fmcs)            
    
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