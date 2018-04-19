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
            fmc = FeatureMCell('FMC'+str(i), self.sensor, self)
            self.fmcs.append(fmc)

        #Init links   
        for i in range(0,FeatureColumn.FMC_NUM):
            posarr = [m for m in range(0,FeatureColumn.FMC_NUM)]
            picksize = random.randrange(round(FeatureColumn.FMC_NUM*0.01), round(FeatureColumn.FMC_NUM*0.03))
            linksPos = random.sample(posarr,picksize)
            fmcs = []
            for pos in linksPos:
                fmcs.append(self.fmcs[pos])
            self.fmcs[i].initFMCLinks(fmcs)            
    
    def run(self):
        for fmc in self.fmcs:
            fmc.run()
            #fmc.debug()
            if (fmc.isFixed):
                print(fmc.output())
        for fmc in self.fmcs:
            fmc.learnSequence()
            fmc.willActive = 0.0

    def getPredictFmc(self):
        for fmc in self.fmcs:
            if fmc.isActive:
                for fmcLink in fmc.fmcLinks:
                    linkedFmc = fmcLink.featuremcell
                    if fmc.isFixed and linkedFmc.isFixed:
                        linkedFmc.willActive = linkedFmc.willActive + fmcLink.weight
        predMap = [fmc.willActive for fmc in self.fmcs]
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

    def output(self):
        print("out")

    def initFmcs(self):
        f = self.fmcs